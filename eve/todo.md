# Sentix Tinder - Master Todo List

*Synthesized from the PRD, Tech Stack, and Design Document (v1.0).*

## Phase 1: Setup & Infrastructure (Week 1)
- [ ] Initialize Git repository and Python virtual environment (`python -m venv .venv`).
- [ ] Install ML dependencies (PyTorch, scikit-learn, HuggingFace, pandas, etc.) and API dependencies (FastAPI, Redis, ONNX).
- [ ] Configure the Kaggle API (`~/.kaggle/kaggle.json`) for data downloading.
- [ ] Setup the React frontend with Vite and Tailwind CSS.
- [ ] Prepare Docker/Docker Compose configurations for local development.

## Phase 2: Data Collection & Preparation (Week 1-2)
- [ ] Scrape 15,000+ total reviews (5,000 each from Tinder, Bumble, Hinge) using `google-play-scraper`.
- [ ] Store raw reviews into `raw_reviews.csv` and configure DVC (Data Version Control).
- [ ] Build the text preprocessing pipeline: lowercase, URL/HTML removal, emoji handling, and punctuation stripping.
- [ ] Integrate NLTK (stopwords) and spaCy `en_core_web_sm` (lemmatization).
- [ ] Filter out sparse reviews (< 5 tokens) and output to `clean_reviews.csv`.

## Phase 3: Classical ML & Baselines (Week 3)
- [ ] Extract TF-IDF feature matrices (`max_features=50,000`).
- [ ] Train baseline models: Multinomial Naïve Bayes, Logistic Regression, and LinearSVC.
- [ ] Evaluate baselines using `classification_report`, `confusion_matrix`, and ROC AUC. Log metrics via MLflow.

## Phase 4: Applied Deep Learning / BERT (Week 4-5)
- [ ] Set up Kaggle Notebook (GPU enabled) or Google Colab for training.
- [ ] Configure HuggingFace Transformers and PyTorch `DataLoader` for review batches.
- [ ] Fine-tune `bert-base-uncased` (Sequence Classification head) for 3-class sentiment (Pos/Neu/Neg).
- [ ] Validate and hit >= 0.78 macro F1 score on the evaluation set.
- [ ] Export the fine-tuned model to ONNX format for rapid, CPU-friendly API inference.

## Phase 5: Advanced Analytics & Extra Features (Week 5-6)
- [ ] Implement Aspect-Based Sentiment Analysis (ABSA) across UI/UX, Pricing, Matches, Bugs, and Safety.
- [ ] Develop Rating vs. Sentiment Mismatch detector (e.g., 5★ rating but predicting Negative sentiment).
- [ ] Build Fake/Spam review detection engine via TF-IDF cosine similarity (>0.90) and repeating-word heuristics.
- [ ] Generate keyword clusters and TF-IDF term frequencies for Word Clouds using NLTK `FreqDist`.
- [ ] (Optional) Implement NLG auto-summary logic for the Compare tab.

## Phase 6: Backend API Development (FastAPI) (Week 7)
- [ ] Initialize FastAPI app with Pydantic type validation and Uvicorn server setup.
- [ ] Load the ONNX BERT model and classical ML pipelines into memory at startup.
- [ ] Implement Dashboard aggregate endpoints (`/api/dashboard/summary`, `/api/dashboard/trends`, `/api/mismatches`).
- [ ] Implement UI data endpoints (`/api/reviews`, `/api/aspects`, `/api/keywords`, `/api/compare`).
- [ ] Develop `/api/predict/batch` endpoint and integrate Celery + Redis for handling async CSV uploads.

## Phase 7: Frontend Interface (React + Tailwind) (Week 8)
- [ ] Configure Tailwind CSS with official aesthetic tokens (`tinder.pink: #FE3C72`, `tinder.orange: #FF7854`, `sentiment.positive: #22C55E`).
- [ ] Setup global application layout enforcing empty states and Mobile First constraints (390px target).
- [ ] Apply **Dark v2 (Glassmorphism)** theme with fixed backdrop-blur Bottom Navigation.
- [ ] **Screen 1 (Dashboard)**: Build KPI stat grid, Sentiment Donut (Chart.js), Trend Line Chart, and actionable Mismatch Alert banner.
- [ ] **Screen 2 (Aspect Analysis)**: Create aspect breakdown metrics with smooth, 3-segment staggered DOM animations.
- [ ] **Screen 3 (Review Explorer)**: Build the filterable Review Cards list, highlighting rating mismatches with amber borders and "WARNING" pill badges.
- [ ] **Screen 4 (Keyword Insights)**: Integrate `react-wordcloud` (or d3-cloud) for Interactive Word Clouds and Top Keyword ranked stat tracks.

## Phase 8: Polish, Testing & Submission (Week 9-10)
- [ ] Write integration and unit tests (`pytest`) covering the FastAPI API endpoints and core NLP preprocessing code.
- [ ] Run Lighthouse accessibility audits targetting >= 90 score.
- [ ] Ensure Python formatting and linting conventions (Black, isort) are passed.
- [ ] Complete final README, complete the GitHub repo documentation, and prepare the demo presentation video.
