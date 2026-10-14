"""Tornado onset prediction from radar.

The question: given the radar scans observed so far, will a tornado appear in the next 5 and 10
minutes when none is visible yet? The model is the published TorNet backbone (Hugging Face,
``tornet-ml/tornado_detector_baseline_v1``) fine-tuned on that target, with a radar forecaster
decoder as an auxiliary training signal and the frozen published detector as the judge that sets
the persistence floor and the ceiling.
"""

__version__ = "0.1.0"
