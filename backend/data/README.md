# Built-in corpus

`tiny_shakespeare.txt` is the first **50,000 Unicode characters** of the real
Tiny Shakespeare corpus from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt),
downloaded on 2026-10-10. Shakespeare's underlying works are public domain.
This committed excerpt needs no network access at training time.

UTF-8 SHA-256: `ef21ba4cfe77713f14d2b6d009ec902a300a9ce33c0a67139c454f03b4e6c968`.
The default chronological split uses the first 45,000 characters for training
and the last 5,000 for validation. Each side produces independent, non-overlapping
input windows and one-character-shifted targets. No window crosses the split.
The vocabulary is constructed from training text only; unseen validation characters
map to `<unk>` and their count is included in task metadata. This is a small corpus
experiment, not a general language-model benchmark. Users may also supply their
own plain text (256–2,000,000 characters).

The word mode keeps English words/contractions, numbers and punctuation as tokens;
Chinese phrases use Jieba 0.42.1 precise segmentation with HMM disabled. Whitespace
is omitted and reconstructed for display. The chronological split moves to the
first token beginning after the character boundary, so a word is never cut between
sets. The built-in word split has 45,004 / 4,996 characters, 10,181 / 1,069 tokens,
and 2,170 vocabulary entries including `<unk>`. There are 147 unseen validation
tokens. Word perplexity and character perplexity use different units.

Character vocabulary permits 255 distinct characters. Word vocabulary keeps up to
4,095 most frequent training tokens (lexical tie break); other training/validation
tokens map to `<unk>`. Both are subject to the model parameter and compute budgets. No tokenizer
or language-model weights are learned from the validation text. Jieba's dictionary
only defines word boundaries; the Transformer is trained from random initialization.

## TinyStories excerpt — changed data, CDLA-Sharing-1.0

`tiny_stories.txt` is a modified excerpt of **TinyStoriesV2-GPT4-train.txt** by
Ronen Eldan and Yuanzhi Li, from the official [roneneldan/TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).
The source contains synthetic GPT-4 short stories with simple English vocabulary.
We train our own randomly initialized Transformer; no pretrained model is loaded.

This data remains available under the unmodified **Community Data License Agreement
– Sharing – Version 1.0**, [full license text](https://cdla.dev/sharing-1-0/).
This data license applies to the excerpt separately from the application code.
Please retain this notice, attribution, license link and the provenance JSON when
redistributing this excerpt or modifications of it.

**Changes:** Downloaded only the first 1.5 MiB from pinned revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` on 2026-10-10. Discarded the incomplete
final story; stripped outer story whitespace; replaced `<|endoftext|>` separators
with blank lines; wrote UTF-8 LF. Full upstream URL and byte range are recorded in
`tiny_stories.provenance.json`.

The excerpt contains **1,901 stories / 1,546,303 characters**. SHA-256:
`283f688aefca37579babe78ddb9c73ed19f669cf0ce9e6810e3f17a8d9e5dc7d`.
The 90/10 chronological word split has 332,462 training and 36,946 validation
tokens; at context 64, evaluation scores 36,928 held-out targets. The train-only
vocabulary has 4,096 entries including `<unk>`; 1,025 training and 562 validation
tokens map to `<unk>`. The split does not cross windows between sets. Story
boundaries are blank lines, omitted by word tokenization; a window may span adjacent
stories. This excerpt's validation is a held-out suffix of the source training file,
not the upstream validation file or an independent test set.

## Small self-introduction training examples

`self_introduction.json` contains 24 project-authored question/answer examples,
covering English and Chinese greetings, identity, capabilities and the creator
name **狗头人**, as requested by the project owner. `read_corpus("stories_identity")`
deterministically interleaves these examples with the TinyStories excerpt, cycling
through them after each blank-line-separated story block. The resulting changed
data retains TinyStories attribution and CDLA-Sharing-1.0 above.

The mixed corpus has 1,662,420 characters; SHA-256:
`d4eec53e5e4efac18395bd91c7d85d0bd163dd10609757e3d0112766cc8b1476`.
The word split has 361,902 / 40,229 train/validation tokens, 4,096 vocabulary
entries, and 1,067 / 568 unknown train/validation tokens. Identity examples repeat
in both parts: identity results measure learned small-example reproduction, not
generalization to new questions. Inference only runs the trained Transformer;
it does not read this JSON or dispatch questions to stored answers.
