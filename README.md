# llmcomp

This code runs the experiments for my undergraduate EE thesis at The University of Sydney. The code is heavily reliant on [DeepMind's](https://github.com/google-deepmind/language_modeling_is_compression) language modelling is copmression paper and uses the arithmetic coder therein. We then expand the paper to use [Gruver et al.](https://github.com/ngruver/llmtime)'s paper on time series prediction with foundation models performing zero-shot prediction. We integrate the probability distributions obtained through the time-series predictor from Gruver et al. to predict and compress WiFi 802.11ax signals generated in MATLAB and compare:

- Base transformer trained on WiFi 802.11ax data in compressing the same;
- Llama-2-7b's innate ability to predict WiFi 802.11ax data, and;
- Llama-2-7b's hypothesised improved ability after fine tuning the foundation model.

We utilise Lightning-Ai's [Lit-Llama](https://github.com/Lightning-AI/lit-llama) to fine tune.
