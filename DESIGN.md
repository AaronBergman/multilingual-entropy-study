# Study design

This protocol was fixed before inspecting any model or n-gram outcomes.

## Research questions

1. How strongly do cross-language bits-per-byte rankings agree between byte
   n-grams and a multilingual LLM?
2. Does a larger same-family LLM lower bits per byte across all languages?
3. Does scaling preserve, compress, or reorder cross-language differences?
4. How sensitive is the answer to n-gram order and to language-specific versus
   shared multilingual n-gram training?

## Corpus

The study uses the original public FLORES-200 release because it is directly and
reproducibly downloadable without authentication. FLORES sentences are
professionally translated and aligned, so each language expresses essentially
the same underlying content.

- N-gram training: `dev`, 997 sentences per language
- Evaluation: `devtest`, 1,012 sentences per language
- Normalization: Unicode NFC
- Sentence context: each sentence is evaluated independently

The 13-language panel was chosen for script and family diversity:

| Code | Language | Script | Family |
|---|---|---|---|
| `eng_Latn` | English | Latin | Germanic |
| `spa_Latn` | Spanish | Latin | Romance |
| `rus_Cyrl` | Russian | Cyrillic | Slavic |
| `arb_Arab` | Modern Standard Arabic | Arabic | Semitic |
| `hin_Deva` | Hindi | Devanagari | Indo-Aryan |
| `zho_Hans` | Mandarin Chinese | Han (simplified) | Sinitic |
| `jpn_Jpan` | Japanese | Japanese mixed | Japonic |
| `kor_Hang` | Korean | Hangul | Koreanic |
| `tur_Latn` | Turkish | Latin | Turkic |
| `swh_Latn` | Swahili | Latin | Bantu |
| `yor_Latn` | Yoruba | Latin | Volta-Niger |
| `tam_Taml` | Tamil | Tamil | Dravidian |
| `tha_Thai` | Thai | Thai | Kra-Dai |

## Models

### Byte n-grams

Fixed-order Markov models of orders 1 through 5 predict raw UTF-8 bytes with
add-0.1 smoothing over the 256-byte vocabulary. The primary n-gram is order 5.
Sentence starts receive `order - 1` dedicated BOS symbols; BOS and EOS are not
included in the score.

Two training scopes are reported:

- **monolingual:** one model per language, trained on that language's aligned
  `dev` split;
- **balanced multilingual:** one shared model trained on all 13 aligned `dev`
  splits, with exactly 997 sentences contributed by every language.

The first estimates within-language local predictability. The second is more
structurally analogous to a single multilingual LLM. Neither has training-data
exposure comparable to the pretrained LLMs, so absolute n-gram and LLM loss
levels are not treated as a contest.

### LLMs

- `mlx-community/Qwen3-0.6B-4bit`
- `mlx-community/Qwen3-4B-4bit`

These models share a family, tokenizer design, post-training regime, and
quantization format while differing substantially in parameter count. Each
sentence is prefixed with BOS (or the tokenizer's EOS fallback), and every text
token is scored. Prefix and padding tokens are excluded.

The comparison tests a scale contrast, not a universal law of model quality.
It does not isolate parameter count from every possible size-specific training
difference.

## Estimands

For a language \(l\), model \(m\), tokenized sentence likelihoods are accumulated
as total negative log probability in bits:

\[
\mathrm{BPB}_{l,m} =
\frac{\sum_i -\log_2 p_m(x_{l,i})}
     {\sum_i |\mathrm{UTF8}(x_{l,i})|}.
\]

The primary estimator is a ratio of sums, not the unweighted mean of sentence
ratios. Secondary normalization replaces UTF-8 bytes with Python Unicode code
points, labeled “characters” rather than grapheme clusters.

Reported comparisons:

- Spearman rank correlation and Kendall's tau across 13 languages;
- per-language larger-minus-smaller LLM BPB;
- cross-language mean, standard deviation, minimum, and maximum BPB;
- ranks and within-metric z-scores;
- tokens per byte and tokens per character for tokenizer diagnostics.

## Uncertainty

Aggregate BPB and paired model-size deltas receive delete-one-sentence
jackknife standard errors and normal 95% intervals. This deterministic choice
avoids arbitrary random seeds. Correlations across only 13 languages are
reported with conventional p-values but interpreted descriptively.

## Falsification and sensitivity checks

- All 1,012 aligned evaluation IDs must exist exactly once for every
  language-model combination.
- Tokenizer encode/decode round trips are counted and reported.
- N-gram order 1–5 trajectories reveal sparse-context sensitivity.
- Monolingual and shared multilingual n-gram variants reveal training-scope
  sensitivity.
- Bits per character is compared with bits per byte.
- Tokenizer fertility is plotted against measured LLM loss.

## Claims this design cannot support

- that one natural language is intrinsically more complex than another;
- that Qwen3 results generalize to all LLM families or tokenizers;
- that scaling alone caused every observed difference;
- that 4-bit likelihoods exactly equal full-precision likelihoods;
- that FLORES translationese represents everyday language use; or
- that UTF-8 bytes or Unicode code points are linguistically neutral units.
