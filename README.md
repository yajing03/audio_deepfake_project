# Momenta - Audio Deepfake Detection Take-Home Assessment

## Part 1: Research & Selection

### 1.M2S-ADD: Mono-to-Stereo Audio Deepfake Detection

#### Betray Oneself: A Novel Audio DeepFake Detection Model via Mono-to-Stereo Conversion

- Key Technical Innovation:
    - Converts mono audio into stereo using a pretrained synthesizer, then applies a dual-branch architecture to detect inconsistencies between left/right channels.

- Reported Performance:
    - Outperforms all mono-based baselines on the ASVspoof 2019 dataset.

- Why Promising:
    - Exposes deepfake artifacts in a novel stereo space
    - Lightweight, efficient model → ideal for real-time detection
    - Performs well on real conversational speech

- Potential Limitations:
    - Preprocessing step (mono-to-stereo) adds complexity
    - May require noise adaptation in practical environments

### 2. RawNet3 + Self-Attentive Pooling

#### RawNet3: Neural Network Architecture for End-to-End Spoofing Detection from Raw Waveform

- Key Technical Innovation:
    - End-to-end deep model that takes raw waveform input, avoiding handcrafted features. Uses self-attentive pooling to focus on informative time segments.

- Reported Performance:
    - Achieves ~1.33% Equal Error Rate (EER) on ASVspoof 2019 LA.

- Why Promising:
    - Minimal preprocessing → fast pipeline
    - Effective in detecting synthetic artifacts in raw speech
    - Real-time capable and suitable for deployment in audio stream settings

- Potential Limitations:

    - Generalization to out-of-domain fakes may require tuning
    - Slight performance drop in noisy conditions

### 3. SONAR: Synthetic Audio Detection Framework & Benchmark

#### SONAR: A Synthetic AI-Audio Detection Framework and Benchmark

- Key Technical Innovation:
    - Introduces a new benchmark dataset from 9 audio synthesis systems. Compares conventional and foundation model-based deepfake detectors.

- Reported Performance:
    - Foundation models (e.g. Whisper) outperform others in generalization to unseen AI-generated speech.

- Why Promising:
    - Emphasizes robust detection in real-world audio
    - Highlights weaknesses in current detection pipelines
    - Encourages use of foundation models with better generalization

- Potential Limitations:
    - Not a detection model itself — it's a benchmarking framework
    - Foundation model inference can be resource-intensive


## Part 2: Implementation

See jupyter notebook: `rawnet2_asvspoof2019_final.ipynb`

## Part 3: Documentation & Analysis

### 1. Implementation Process

Challenges Encountered:

- Incorrect path references in the official baseline code caused file-not-found errors (e.g., missing slashes, hardcoded folder names).

- The original yaml.load() function threw an error due to missing Loader= argument (in newer PyYAML versions).

- Training on CPU was very slow, making it impractical to complete full training or evaluation.

How I Addressed These Challenges:

- Updated the file path logic using os.path.join() properly and matched the folder names to what the script expected.

- Replaced yaml.load(f) with yaml.load(f, Loader=yaml.SafeLoader) to resolve the parsing issue.

- Reduced training epochs and batch size to speed things up on CPU.

Assumptions Made:

- That ASVspoof2019 LA train/dev/eval folders would follow the original structure and naming conventions.

- That using a partial training result and benchmark EER value from literature is acceptable for evaluation, as allowed in the instructions.

### 2. Analysis

Why I Selected RawNet2:

- RawNet2 was available in the official ASVspoof2021 GitHub repo and requires minimal preprocessing (uses raw waveforms).

- It strikes a balance between simplicity, performance, and real-time feasibility — ideal for the detection of AI-generated speech.

How the Model Works:

- RawNet2 is an end-to-end model that takes raw audio waveforms as input.

- It processes these with 1D convolution layers followed by residual blocks and a GRU-based recurrent layer.

- The final fully connected layer outputs logits for two classes: bona fide vs. spoofed audio.

Performance Results:

- Early training showed ~79.38% accuracy on the training data.

- Evaluation was not completed due to long CPU runtime, but literature reports EER ≈ 0.02 for RawNet2 on ASVspoof2019 LA.

Strengths & Weaknesses:

- Strengths:

    - Direct use of raw audio (no feature engineering needed)

    - Small and efficient architecture — potential for real-time detection

    - Strong baseline results on ASVspoof datasets

- Weaknesses:

    - Sensitive to noise or channel mismatches

    - May struggle to generalize to unseen deepfake generation methods

    - Long training times on CPU hardware

Suggestions for Future Improvements:

- Use data augmentation to improve generalization (e.g., noise, compression artifacts)

- Add channel-aware layers or attention modules as in RawNet3

- Optimize for inference via quantization or distillation

- Train with more real-world or conversational audio data

### 3. Reflection Questions

#### 1. What were the most significant challenges in implementing this model?

- Debugging hardcoded paths and YAML loading issues in the baseline code.

- Managing training time on CPU — had to limit epochs and batch size.

#### 2. How might this approach perform in real-world conditions vs. research datasets?

- Performance may drop in noisy, phone-quality, or real-time environments.

- Generalization to unseen synthesis methods may require further tuning and data diversity.

#### 3. What additional data or resources would improve performance?

- Diverse deepfake samples from multiple synthesis tools

- Real-world conversations with background noise or compression

- GPU access to enable full training and hyperparameter tuning

#### 4. How would you approach deploying this model in a production environment?

- Convert to TorchScript or ONNX for real-time deployment

- Wrap inference in a lightweight API

- Include preprocessing for streaming audio (e.g., chunking)

- Continuously retrain with real-world feedback data