import subprocess
from datetime import datetime, timezone

import pytest

from minuteforge.audio import (
    CHANNELS,
    SAMPLE_RATE,
    AudioError,
    extract_audio,
    recorded_at,
)


class FakeFfmpeg:
    """Заглушка вместо ffmpeg: запоминает команду и создаёт файл."""

    def __init__(self, returncode=0, stderr="", create=True):
        self.returncode = returncode
        self.stderr = stderr
        self.create = create
        self.commands = []

    def __call__(self, command):
        self.commands.append(list(command))
        if self.create and self.returncode == 0:
            target = command[-1]
            open(target, "wb").close()
        return subprocess.CompletedProcess(command, self.returncode, "", self.stderr)


def found(name):
    return "/usr/bin/ffmpeg"


def missing(name):
    return None


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"not really a video")
    return path


def test_audio_is_mono_16k_as_whisper_expects(video):
    """Другая частота означает лишний пересчёт внутри модели."""
    ffmpeg = FakeFfmpeg()
    extract_audio(video, run=ffmpeg, which=found)

    command = ffmpeg.commands[0]
    assert command[:2] == ["ffmpeg", "-y"]
    assert command[command.index("-ar") + 1] == str(SAMPLE_RATE)
    assert command[command.index("-ac") + 1] == str(CHANNELS)
    assert "-vn" in command, "видеодорожка распознаванию не нужна"


def test_target_defaults_to_the_video_name(video):
    result = extract_audio(video, run=FakeFfmpeg(), which=found)
    assert result == video.with_suffix(".wav")
    assert result.exists()


def test_loudness_is_normalized_by_default(video):
    """Председатель у микрофона и участник в конце зала отличаются на
    десятки децибел; тихого распознавание не слышит."""
    ffmpeg = FakeFfmpeg()
    extract_audio(video, run=ffmpeg, which=found)
    assert "loudnorm" in " ".join(ffmpeg.commands[0])


def test_normalization_can_be_turned_off(video):
    ffmpeg = FakeFfmpeg()
    extract_audio(video, normalize=False, run=ffmpeg, which=found)
    assert "loudnorm" not in " ".join(ffmpeg.commands[0])


def test_existing_audio_is_not_redone(video):
    """Раскодирование часовой записи занимает минуты."""
    video.with_suffix(".wav").write_bytes(b"already done")
    ffmpeg = FakeFfmpeg()
    extract_audio(video, run=ffmpeg, which=found)
    assert ffmpeg.commands == [], "ffmpeg вызывать не следовало"


def test_overwrite_forces_the_work(video):
    video.with_suffix(".wav").write_bytes(b"stale")
    ffmpeg = FakeFfmpeg()
    extract_audio(video, overwrite=True, run=ffmpeg, which=found)
    assert len(ffmpeg.commands) == 1


def test_missing_ffmpeg_tells_how_to_install_it(video):
    with pytest.raises(AudioError, match="brew install ffmpeg"):
        extract_audio(video, run=FakeFfmpeg(), which=missing)


def test_missing_video_is_caught_before_ffmpeg(tmp_path):
    ffmpeg = FakeFfmpeg()
    with pytest.raises(AudioError, match="не найден"):
        extract_audio(tmp_path / "нет.mp4", run=ffmpeg, which=found)
    assert ffmpeg.commands == []


def test_ffmpeg_failure_shows_the_tail_of_its_output(video):
    """ffmpeg пишет десятки строк о кодеках, а причина всегда в конце."""
    noise = "\n".join(f"строка про кодек {i}" for i in range(30))
    ffmpeg = FakeFfmpeg(returncode=1, stderr=f"{noise}\nInvalid data found when processing input")
    with pytest.raises(AudioError, match="Invalid data found"):
        extract_audio(video, run=ffmpeg, which=found)


def test_silent_failure_is_caught(video):
    """ffmpeg вернул ноль, а файла нет — молча это пропускать нельзя."""
    ffmpeg = FakeFfmpeg(create=False)
    with pytest.raises(AudioError, match="файла нет"):
        extract_audio(video, run=ffmpeg, which=found)


def test_target_directory_is_created(video, tmp_path):
    target = tmp_path / "выгрузка" / "audio.wav"
    extract_audio(video, target, run=FakeFfmpeg(), which=found)
    assert target.exists()


class FakeProbe:
    """Заглушка вместо ffprobe: отдаёт заготовленный ответ."""

    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode
        self.commands = []

    def __call__(self, command):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, "")


def test_the_recording_start_is_read_from_the_container(video):
    """Программы видеосвязи пишут в контейнер момент начала записи. Для
    совещания это и есть его время — в отличие от времени файла на диске."""
    probe = FakeProbe("2026-09-07T07:00:12.000000Z\n")
    moment = recorded_at(video, run=probe, which=found)

    # По Гринвичу, поэтому переводится в местное: в протоколе нужен тот час,
    # который участники видели на своих часах.
    expected = datetime(2026, 9, 7, 7, 0, 12, tzinfo=timezone.utc).astimezone()
    assert moment == expected.replace(tzinfo=None)
    assert "creation_time" in " ".join(probe.commands[0])


def test_a_recording_without_the_mark_is_not_given_one(video):
    """Метки нет — это обычное дело, а не ошибка: дату впишут руками."""
    assert recorded_at(video, run=FakeProbe(""), which=found) is None
    assert recorded_at(video, run=FakeProbe("N/A\n"), which=found) is None
    # Кодировщик, которому время неизвестно, пишет начало эпохи.
    assert recorded_at(video, run=FakeProbe("1970-01-01T00:00:00.000000Z"), which=found) is None
    assert recorded_at(video, run=FakeProbe("", returncode=1), which=found) is None
    assert recorded_at(video, run=FakeProbe("2026-09-07T07:00:12Z"), which=missing) is None
