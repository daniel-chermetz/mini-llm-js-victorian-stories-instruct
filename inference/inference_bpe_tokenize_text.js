globalThis.tokenizeText = (textByWords, trainedVocab) => {
	const tokenizedText = [];

	for (let wordIndex = 0; wordIndex < textByWords.length; wordIndex++) {
		let word = textByWords[wordIndex].split('');
		let wordMerged = false;
		let nextIterationWordStaging = [];
		let nextMergeIndex = -1;
		let nextMergePriority = 50000; // lower is better

		while (!wordMerged) {
			wordMerged = true;
			nextMergeIndex = -1;
			nextMergePriority = 50000;

			for (let subWordIndex = 0; subWordIndex < word.length; subWordIndex++) {
				let firstPart = word[subWordIndex];

				let secondPart;
				if (subWordIndex < word.length - 1) {
					secondPart = word[subWordIndex + 1];

					for (let i = 0; i < trainedVocab.length; i++) {
						const currentToken =  trainedVocab[i];
						if (currentToken[0] === firstPart && currentToken[1] === secondPart) {
							if (i < nextMergePriority) {
								wordMerged = false;
								nextMergePriority = i;
								nextMergeIndex = subWordIndex;
							}
						}
					}				
				}
			}

			for (let subWordIndex = 0; subWordIndex < word.length; subWordIndex++) {
				if (subWordIndex === nextMergeIndex) {
					nextIterationWordStaging.push(trainedVocab[nextMergePriority][2]);
					subWordIndex++;
				} else {
					nextIterationWordStaging.push(word[subWordIndex]);
				}
			}

			word = nextIterationWordStaging;
			nextIterationWordStaging = [];
		}

		tokenizedText.push(...word);
	}

	return tokenizedText;
}

// Stage 2: validate every token produced by tokenizeText against the model's
// actual vocabulary (victorianVocab.json). Any token that isn't present is
// decomposed back into the two antecedent tokens it was merged from (looked up
// in the BPE merge vocab, victorianBPEVocab.json) and re-checked recursively.
// A single character that isn't in the vocabulary is replaced with '~'.
//   tokenizedText  : array of token strings (output of tokenizeText)
//   bpeVocab       : array of [part1, part2, merged] merge rules
//   modelVocab     : array of valid token strings (the model vocabulary)
// Returns a new array of token strings, all guaranteed to exist in modelVocab
// (except '~', which itself is part of the base vocabulary).
globalThis.validateTokensAgainstVocab = (tokenizedText, bpeVocab, modelVocab) => {
	const modelVocabSet = new Set(modelVocab);

	// Map a merged token string back to its two antecedent parts.
	const mergeMap = {};
	for (let i = 0; i < bpeVocab.length; i++) {
		const part1 = bpeVocab[i][0];
		const part2 = bpeVocab[i][1];
		const merged = bpeVocab[i][2];
		// Keep the first (highest-priority) rule that produced this merged token.
		if (!(merged in mergeMap)) {
			mergeMap[merged] = [part1, part2];
		}
	}

	const result = [];

	const resolveToken = (token) => {
		// Already a valid vocabulary token.
		if (modelVocabSet.has(token)) {
			result.push(token);
			return;
		}

		// Known merged token: split into its two antecedents and recheck each.
		if (token in mergeMap) {
			resolveToken(mergeMap[token][0]);
			resolveToken(mergeMap[token][1]);
			return;
		}

		// Unknown multi-character token with no merge rule: break it down to its
		// individual characters and resolve each one.
		const chars = Array.from(token);
		if (chars.length > 1) {
			for (let i = 0; i < chars.length; i++) {
				resolveToken(chars[i]);
			}
			return;
		}

		// Single character that isn't in the vocabulary.
		result.push('~');
	};

	for (let i = 0; i < tokenizedText.length; i++) {
		resolveToken(tokenizedText[i]);
	}

	return result;
}

// Stage 3: orchestration. Takes raw custom text and returns an array of token
// strings (not indices) that are all valid entries of the model vocabulary.
//   1. Split the text into words (divideTextIntoWords).
//   2. Load the BPE merge vocab (victorianBPEVocab.json) and the model vocab
//      (victorianVocab.json), caching both on globalThis.
//   3. Tokenize the words with the BPE merge vocab (tokenizeText).
//   4. Validate / decompose the tokens against the model vocab
//      (validateTokensAgainstVocab).
globalThis.tokenizeCustomText = async (text) => {
	// 0. Normalize the raw text (dash spacing, single-quote standardization)
	//    before it is split into words.
	const normalizedText = globalThis.normalizeText(text);

	// 1. Split into words.
	const words = globalThis.divideTextIntoWords(normalizedText);

	// 2. Load and cache the BPE merge vocab.
	if (!globalThis.bpeVocab) {
		const bpeResponse = await fetch(NetworkMeta.CONFIG_VOCAB_V2 ? './model/vocab/victorianBPEVocab_V2.json' : './model/vocab/victorianBPEVocab.json');
		globalThis.bpeVocab = await bpeResponse.json();
	}
	const bpeVocab = globalThis.bpeVocab;

	// Load and cache the model vocab (reusing globalThis.vocab if already loaded).
	if (!globalThis.vocab) {
		const vocabResponse = await fetch(NetworkMeta.CONFIG_VOCAB_V2 ? './model/vocab/victorianVocab_V2.json' : './model/vocab/victorianVocab.json');
		globalThis.vocab = await vocabResponse.json();
	}
	const modelVocab = globalThis.vocab;

	// 3. Tokenize the words using the BPE merge vocab.
	const rawTokens = globalThis.tokenizeText(words, bpeVocab);

	// 4. Validate against the model vocab, decomposing any unknown tokens.
	return globalThis.validateTokensAgainstVocab(rawTokens, bpeVocab, modelVocab);
}