import sys
import types

import torch


class DummyBuilder:
    def __init__(self, data):
        self._data = data

    def and_return(self):
        return iter(self._data)


def _install_fake_fairseq(monkeypatch):
    def _make_module(name):
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package when needed
        sys.modules[name] = mod
        return mod

    fairseq2 = _make_module("fairseq2")
    data = _make_module("fairseq2.data")

    mem = _make_module("fairseq2.data._memory")
    mem.MemoryBlock = type("MemoryBlock", (), {})

    audio = _make_module("fairseq2.data.audio")
    audio.AudioDecoder = type("AudioDecoder", (), {})

    dp = _make_module("fairseq2.data.data_pipeline")
    for cls_name in [
        "CollateOptionsOverride",
        "Collater",
        "DataPipeline",
        "DataPipelineBuilder",
        "FileMapper",
    ]:
        setattr(dp, cls_name, type(cls_name, (), {}))

    def read_sequence(x):
        return x

    dp.read_sequence = read_sequence

    tok = _make_module("fairseq2.data.tokenizers")
    tok.Tokenizer = type("Tokenizer", (), {})

    tok_hub = _make_module("fairseq2.data.tokenizers.hub")
    tok_hub.load_tokenizer = lambda *args, **kwargs: None

    datasets = _make_module("fairseq2.datasets")
    batch = _make_module("fairseq2.datasets.batch")
    batch.Seq2SeqBatch = type("Seq2SeqBatch", (), {})

    logging_mod = _make_module("fairseq2.logging")

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    logging_mod.get_log_writer = lambda *args, **kwargs: DummyLogger()

    models = _make_module("fairseq2.models")
    hub = _make_module("fairseq2.models.hub")
    hub.load_model = lambda *args, **kwargs: None

    w2v2_asr = _make_module("fairseq2.models.wav2vec2.asr")
    w2v2_asr.Wav2Vec2AsrModel = type("Wav2Vec2AsrModel", (), {})

    nn = _make_module("fairseq2.nn")
    batch_layout = _make_module("fairseq2.nn.batch_layout")
    batch_layout.BatchLayout = type("BatchLayout", (), {})
    nn.BatchLayout = batch_layout.BatchLayout

    fairseq2.data = data
    fairseq2.datasets = datasets
    fairseq2.logging = logging_mod
    fairseq2.models = models
    fairseq2.nn = nn

    # Stub wav2vec2-llama modules so pipeline import doesn't pull heavy deps
    llama_pkg = _make_module("omnilingual_asr.models.wav2vec2_llama")
    llama_model = _make_module("omnilingual_asr.models.wav2vec2_llama.model")
    llama_model.Wav2Vec2LlamaModel = type("Wav2Vec2LlamaModel", (), {})
    llama_config = _make_module("omnilingual_asr.models.wav2vec2_llama.config")
    llama_config.ModelType = type("ModelType", (), {"ZERO_SHOT": "zs"})
    llama_beam = _make_module("omnilingual_asr.models.wav2vec2_llama.beamsearch")
    llama_beam.Wav2Vec2LlamaBeamSearchSeq2SeqGenerator = type(
        "Wav2Vec2LlamaBeamSearchSeq2SeqGenerator", (), {}
    )
    llama_model.Wav2Vec2LlamaBeamSearchConfig = type(
        "Wav2Vec2LlamaBeamSearchConfig", (), {}
    )


def test_transcribe_auto_chunks_long_audio(monkeypatch):
    _install_fake_fairseq(monkeypatch)

    from omnilingual_asr.models.inference import pipeline as pipeline_mod

    # Create a 50s mono waveform to force chunking beyond the 40s window
    sample_rate = 16000
    duration_sec = 50
    waveform = torch.zeros(int(sample_rate * duration_sec))

    class FakeCTC(torch.nn.Module):
        model_type = None

    fake_model = FakeCTC()

    # Ensure isinstance checks treat FakeCTC as a Wav2Vec2AsrModel
    monkeypatch.setattr(pipeline_mod, "Wav2Vec2AsrModel", FakeCTC)
    monkeypatch.setattr(pipeline_mod, "Wav2Vec2LlamaModel", FakeCTC)

    def fake_build(self, *args, **kwargs):
        return DummyBuilder([waveform])

    def fake_batch(self, batch_data):
        return batch_data

    def fake_apply(self, batch):
        return ["hello" for _ in batch]

    def fake_align(model, wav_segment, sr, text):
        return [{"word": text, "start": 0.0, "end": len(wav_segment) / sr}]

    monkeypatch.setattr(pipeline_mod, "align_ctc", fake_align)

    pipe = object.__new__(pipeline_mod.ASRInferencePipeline)
    pipe.model = fake_model
    pipe.dtype = torch.float32
    pipe.device = torch.device("cpu")
    pipe._build_audio_wavform_pipeline = types.MethodType(fake_build, pipe)
    pipe._create_batch_simple = types.MethodType(fake_batch, pipe)
    pipe._apply_model = types.MethodType(fake_apply, pipe)

    transcripts, timestamps = pipe.transcribe(["placeholder"], chunk_len=None)

    # Expect two chunks: 40s + 10s
    assert transcripts == ["hello hello"]
    assert len(timestamps) == 1
    assert len(timestamps[0]) == 2
    assert timestamps[0][0]["start"] == 0.0
    assert timestamps[0][0]["end"] == 40.0
    assert timestamps[0][1]["start"] == 40.0
    assert timestamps[0][1]["end"] == 50.0
