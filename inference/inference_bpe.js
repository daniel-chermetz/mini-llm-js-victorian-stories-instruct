// Letters: A–Z and a–z
const letters = {};
"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzéÉçäèâá".split("").forEach(ch => {
  letters[ch] = true;
});

// Digits: 0–9
const digits = {};
"0123456789".split("").forEach(ch => {
  digits[ch] = true;
});

// Common punctuation
const punctuation = {};
",.;:!?\"'`()[]{}<>/\\|@#$%^&*_+=~“”‘’".split("").forEach(ch => {
  punctuation[ch] = true;
});

const spacesAndLineBreaks = {};
spacesAndLineBreaks[' '] = true;
spacesAndLineBreaks['\n'] = true;
spacesAndLineBreaks['\t'] = true;

const punctuationThatMustBeItsOwnWord = {};
"—–-…".split("").forEach(ch => {
	punctuationThatMustBeItsOwnWord[ch] = true;
});

// Normalize raw text before it is split into words: ensure a space follows any
// dash (em, en, hyphen-minus) and standardize single quotes to the curly right
// quote (’).
globalThis.normalizeText = (text) => {
	// 1. Space after dashes: \u2014 (em), \u2013 (en), - (hyphen-minus)
	text = text.replace(/([\u2014\u2013-])(?! )/g, '$1 ');
	// 2. Standardize single quotes to curly right \u2019
	text = text.replace(/[\u2018']/g, '\u2019');
	return text;
}

globalThis.divideTextIntoWords = (letterSeq) => {
	const words = [];

	let currentWord = '';
	for (let i = 0; i < letterSeq.length; i++) {
		let currentLetter = letterSeq[i];
		// previously I've added this 'if' statement but haven't had the !spacesAndLineBreaks[currentLetter] case
		if (!spacesAndLineBreaks[currentLetter] && !letters[currentLetter] && !digits[currentLetter] && !punctuation[currentLetter] && !punctuationThatMustBeItsOwnWord[currentLetter]) {
			currentLetter = '~';
		}

		let wordEnd = false;

		if (!currentWord.length) {
			currentWord = currentWord + currentLetter;
			continue;
		}

		if (punctuationThatMustBeItsOwnWord[currentLetter]) {
			// dashes should always be their own word
			wordEnd = true;
		}

		// encountered punctuation; should stop accumulated word?
		if (punctuation[currentLetter] && currentWord.at(-1) !== ' ') {
			if (i === letterSeq.length - 1) {
				// at the end of the whole sequence
				wordEnd = true;
			}
			if (i < (letterSeq.length - 1) && (!letters[letterSeq[i + 1]] && !digits[letterSeq[i + 1]])) {
				// there's another punctuation right after the current one
				wordEnd = true;
			}
			if (!letters[currentWord.at(-1)] && !digits[currentWord.at(-1)]) {
				// there's another punctuation as the last character in the accumulated word
				wordEnd = true;
			}
			if (currentLetter === '.' && (currentWord === '.' || currentWord === ' .' || currentWord === '..' || currentWord === ' ..')) {
				// override: if the preceding is a dot we can accumulate up to 3 dots to make an ellipsis
				wordEnd = false;
			}
			if (currentLetter === '*' && (currentWord === '*' || currentWord === ' *')) {
				// override: if the preceding is a * we can accumulate up to 2 stars
				wordEnd = false;
			}			
		}

		// encountered letter or digit; should accumulate into word?
		if (letters[currentLetter] || digits[currentLetter]) {
			if (currentWord.length === 1 && (punctuation[currentWord[0]] || currentWord[0] === '\n' || currentWord[0] === '\t')) {
				// the accumulated word is a single punctuation mark, and the new letter should be starting a new word
				wordEnd = true;
			}
			if (currentWord.length === 2 && currentWord[0] === ' ' && punctuation[currentWord[1]]) {
				// the accumulated word is space followed by punctuation mark, and the new letter should be starting a new word
				wordEnd = true;
			}
			if (currentWord.length >= 2 && punctuation[currentWord.at(-1)] && punctuation[currentWord.at(-2)]) {
				// the accumulated word is either '..', '**', ' ..', or ' **', and the new letter should be starting a new word
				wordEnd = true;
			}			
		}

		if (currentLetter === ' ') {
			// a space breaks the accumulated word, to start a new word
			wordEnd = true;
		}
		if (currentLetter === '\n') {
			// a new line breaks the accumulated word, to start a new word
			wordEnd = true;
		}
		if (currentLetter === '\t') {
			// a tab breaks the accumulated word, to start a new word
			wordEnd = true;
		}

		if (punctuationThatMustBeItsOwnWord[currentWord.at(-1)]) {
			// certain punctuation marks should always stand on their own
			wordEnd = true;
		}

		if (wordEnd) {
			words.push(currentWord);
			currentWord = '';
			i--;
		} else {
			currentWord += currentLetter;
		}
	}
	if (currentWord.length) {
		words.push(currentWord);
	}

	return words;
}

const vocab = [];
globalThis.getVocab = () => {
	return vocab;
}

globalThis.countTokenCombos = (words) => {
	const tokenCombosCount = {};
	const tokenComboSubStrings = {};

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			let part1 = currentWord[subWordIndex];
			if (Array.isArray(part1)) {
				part1 = part1[0];
			}
			let part2 = currentWord[subWordIndex + 1];
			if (Array.isArray(part2)) {
				part2 = part2[0];
			}

			const comboStr = part1 + part2;
			if (!tokenComboSubStrings[comboStr]) {
				tokenComboSubStrings[comboStr] = [part1, part2];
			}
			if (tokenCombosCount[comboStr]) {
				tokenCombosCount[comboStr]++
			} else {
				tokenCombosCount[comboStr] = 1;
			}
		}
	}

	return [tokenCombosCount, tokenComboSubStrings];
}

globalThis.countTokenCombosV2 = (words) => {
	const tokenCombosCount = {};
	const tokenComboSubStrings = {};

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex += 1) {
			let part1EndIndex = subWordIndex;
			let part1 = currentWord[subWordIndex];
			if (part1[0] === '{') {
				part1 = '';
				for (let iterator = subWordIndex + 1; iterator < currentWord.length; iterator++) {
					if (currentWord[iterator] === '}') {
						part1EndIndex = iterator;
						subWordIndex = iterator;
						break;
					}
					part1 += currentWord[iterator];
				}
			}

			if (part1EndIndex < currentWord.length - 1) {
				const part2StartIndex = part1EndIndex + 1;
				let part2 = currentWord[part2StartIndex];
				if (part2 === '{') {
					part2 = '';

					for (let iterator = part2StartIndex + 1; iterator < currentWord.length; iterator++) {
						if (currentWord[iterator] === '}') {
							break;
						}

						part2 += currentWord[iterator];
					}
				}

				const comboStr = part1 + part2;
				if (!tokenComboSubStrings[comboStr]) {
					tokenComboSubStrings[comboStr] = [part1, part2];
				}
				if (tokenCombosCount[comboStr]) {
					tokenCombosCount[comboStr]++
				} else {
					tokenCombosCount[comboStr] = 1;
				}
			}
		}
	}

	return [tokenCombosCount, tokenComboSubStrings];
}

// position: LEFT, RIGHT, ANY
globalThis.countTokenCombosWithTargetToken = (words, targetTokenStr, targetTokenPosition = 'ANY') => {
	const tokenCombosCount = {};
	const tokenComboSubStrings = {};

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex += 1) {
			let target = currentWord[subWordIndex];
			if (typeof target !== 'string') {
				target = target[0];
			}
			if (target === targetTokenStr) {
				if (targetTokenPosition === 'RIGHT' || targetTokenPosition === 'ANY') {				
					if (subWordIndex > 0) {
						let onePrior = currentWord[subWordIndex - 1];
						if (typeof onePrior !== 'string') {
							onePrior = onePrior[0];
						}

						const comboStr1 = onePrior + target;
						if (!tokenComboSubStrings[comboStr1]) {
							tokenComboSubStrings[comboStr1] = [onePrior, target];
						}
						if (tokenCombosCount[comboStr1]) {
							tokenCombosCount[comboStr1]++
						} else {
							tokenCombosCount[comboStr1] = 1;
						}
					}
				}
				if (targetTokenPosition === 'LEFT' || targetTokenPosition === 'ANY') {
					if (subWordIndex < currentWord.length - 1) {
						let oneAfter = currentWord[subWordIndex + 1];
						if (typeof oneAfter !== 'string') {
							oneAfter = oneAfter[0];
						}
						const comboStr2 = target + oneAfter;
						if (!tokenComboSubStrings[comboStr2]) {
							tokenComboSubStrings[comboStr2] = [target, oneAfter];
						}		
						if (tokenCombosCount[comboStr2]) {
							tokenCombosCount[comboStr2]++
						} else {
							tokenCombosCount[comboStr2] = 1;
						}
					}
				}
			}
		}
	}

	return [tokenCombosCount, tokenComboSubStrings];
}

globalThis.countTokenCombosWithTargetTokenV2 = (words, targetTokenStr, leftTargetSubStr, rightTargetSubStr) => {
	const tokenCombosCount = {};
	const tokenComboSubStrings = {};

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex += 1) {
			let target = currentWord[subWordIndex];
			if (typeof target !== 'string') {
				target = target[0];
			}
			if (target === targetTokenStr || target === leftTargetSubStr) {				
				if (subWordIndex > 0) {
					let onePrior = currentWord[subWordIndex - 1];
					if (typeof onePrior !== 'string') {
						onePrior = onePrior[0];
					}

					const comboStr1 = onePrior + target;
					if (!tokenComboSubStrings[comboStr1]) {
						tokenComboSubStrings[comboStr1] = [onePrior, target];
					}
					if (tokenCombosCount[comboStr1]) {
						tokenCombosCount[comboStr1]++
					} else {
						tokenCombosCount[comboStr1] = 1;
					}
				}
			}
			if (target === targetTokenStr || target === rightTargetSubStr) {
				if (subWordIndex < currentWord.length - 1) {
					let oneAfter = currentWord[subWordIndex + 1];
					if (typeof oneAfter !== 'string') {
						oneAfter = oneAfter[0];
					}
					const comboStr2 = target + oneAfter;
					if (!tokenComboSubStrings[comboStr2]) {
						tokenComboSubStrings[comboStr2] = [target, oneAfter];
					}		
					if (tokenCombosCount[comboStr2]) {
						tokenCombosCount[comboStr2]++
					} else {
						tokenCombosCount[comboStr2] = 1;
					}
				}
			}
		}
	}

	return [tokenCombosCount, tokenComboSubStrings];
}

// right is original right, but merges when it stands on the LEFT side and looks right
// left is original left, but merges when it stands on the RIGHT side and looks left
// rightTargetSubStr, leftTargetSubStr // dontLookBack!
// rightTargetSubStr, targetTokenStr // dontLookBack!
// leftTargetSubStr, rightTargetSubStr // no issue
// leftTargetSubStr, targetTokenStr // no issue
// targetTokenStr, targetTokenStr // dontLookBack!
// targetTokenStr, leftTargetSubStr // dontLookBack!
// targetTokenStr, rightTargetSubStr // no issue
globalThis.countTokenCombosWithTargetTokenV3 = (words, targetTokenStr, leftTargetSubStr, rightTargetSubStr) => {
	const tokenCombosCount = {};
	const tokenComboSubStrings = {};

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];
		if (currentWord[0] === '{' && currentWord.indexOf('}') === (currentWord.length - 1)) {
			continue;
		}

		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex += 1) {
			const targetStartIndex = subWordIndex;
			let targetEndIndex = subWordIndex;
			let target = currentWord[subWordIndex];
			if (target[0] === '{') {
				target = '';
				for (let iterator = subWordIndex + 1; iterator < currentWord.length; iterator++) {
					if (currentWord[iterator] === '}') {
						targetEndIndex = iterator;
						subWordIndex = iterator;
						break;
					}
					target += currentWord[iterator];
				}
			}

			if (target === targetTokenStr || target === leftTargetSubStr) {
				if (targetStartIndex > 0) {
					const onePriorEndIndex = targetStartIndex - 1;
					let onePrior = currentWord[onePriorEndIndex];
					if (onePrior === '}') {
						onePrior = '';
						for (let iterator = onePriorEndIndex - 1; iterator >= 0; iterator--) {
							if (currentWord[iterator] === '{') {
								break;
							}
							onePrior = currentWord[iterator] + onePrior;
						}
					}

					let dontLookBack = false;
					if (target === targetTokenStr && onePrior === targetTokenStr) {
						dontLookBack = true;
					} else if (target === targetTokenStr && onePrior === rightTargetSubStr) {
						dontLookBack = true;
					} else if (target === leftTargetSubStr && onePrior === targetTokenStr) {
						dontLookBack = true;
					} else if (target === leftTargetSubStr && onePrior === rightTargetSubStr) {
						dontLookBack = true;
					}

					if (!dontLookBack) {
						const comboStr1 = onePrior + target;
						if (!tokenComboSubStrings[comboStr1]) {
							tokenComboSubStrings[comboStr1] = [onePrior, target];
						}
						if (tokenCombosCount[comboStr1]) {
							tokenCombosCount[comboStr1]++
						} else {
							tokenCombosCount[comboStr1] = 1;
						}
					}
				}
			}
			if (target === targetTokenStr || target === rightTargetSubStr) {
				if (targetEndIndex < currentWord.length - 1) {
					const oneAfterStartIndex = targetEndIndex + 1;
					let oneAfter = currentWord[oneAfterStartIndex];
					if (oneAfter === '{') {
						oneAfter = '';

						for (let iterator = oneAfterStartIndex + 1; iterator < currentWord.length; iterator++) {
							if (currentWord[iterator] === '}') {
								break;
							}

							oneAfter += currentWord[iterator];
						}
					}

					const comboStr2 = target + oneAfter;
					if (!tokenComboSubStrings[comboStr2]) {
						tokenComboSubStrings[comboStr2] = [target, oneAfter];
					}		
					if (tokenCombosCount[comboStr2]) {
						tokenCombosCount[comboStr2]++
					} else {
						tokenCombosCount[comboStr2] = 1;
					}
				}
			}
		}
	}

	return [tokenCombosCount, tokenComboSubStrings];
}

globalThis.getMostFrequentTokenCombo = (tokenCombosCount) => {
	let mostFrequentComboStr = "";
	let mostFrequentComboCount = 0;
	for (let comboStr in tokenCombosCount) {
		if (tokenCombosCount[comboStr] > mostFrequentComboCount) {
			mostFrequentComboStr = comboStr;
			mostFrequentComboCount = tokenCombosCount[comboStr];
		}
	}

	return mostFrequentComboStr;
}

globalThis.addNextMostFrequentTokenToVocab = (mostFrequentTokenComboStr) => {
	vocab.push(mostFrequentTokenComboStr);
}

globalThis.mergePerNextHighestPriorityTokens = (targetTokenComboStr, words) => {
	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			let part1 = currentWord[subWordIndex];
			if (Array.isArray(part1)) {
				part1 = part1[0];
			}
			let part2 = currentWord[subWordIndex + 1];
			if (Array.isArray(part2)) {
				part2 = part2[0];
			}

			if (part1 + part2 === targetTokenComboStr) {
				let wordPostMerge = [];
				for (let i = 0; i < currentWord.length; i++) {
					if (i < subWordIndex) {
						wordPostMerge.push(currentWord[i]);
					}
					if (i > subWordIndex + 1) {
						wordPostMerge.push(currentWord[i]);
					}
					if (i === subWordIndex) {
						wordPostMerge.push([part1 + part2]);
					}
				}

				words[wordIndex] = wordPostMerge;
				wordIndex--; // check same word again for another repition of the same combo
				break;
			}
		}
	}

	return words;
}

globalThis.mergePerNextHighestPriorityTokensV2 = (leftTargetSubStr, rightTargetSubStr, words) => {
	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];

		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			let part1 = currentWord[subWordIndex];
			if (typeof part1 !== 'string') {
				part1 = part1[0];
			}
			if (part1 !== leftTargetSubStr) {
				continue;
			}
			let part2 = currentWord[subWordIndex + 1];
			if (typeof part2 !== 'string') {
				part2 = part2[0];
			}
			if (part2 !== rightTargetSubStr) {
				continue;
			}

			let wordPostMerge = [];
			for (let i = 0; i < currentWord.length; i++) {
				if (i < subWordIndex) {
					wordPostMerge.push(currentWord[i]);
				}
				if (i > subWordIndex + 1) {
					wordPostMerge.push(currentWord[i]);
				}
				if (i === subWordIndex) {
					wordPostMerge.push([part1 + part2]);
				}
			}

			words[wordIndex] = wordPostMerge;
			wordIndex--; // check same word again for another repition of the same combo
			break; // need to break because the length of the currentWord changed with the merge
		}
	}

	return words;
}

globalThis.mergePerNextHighestPriorityTokensV3 = (leftTargetSubStr, rightTargetSubStr, words) => {
	const finalMergedWord = leftTargetSubStr + rightTargetSubStr;
	let lengthOfOriginalTokens = finalMergedWord.length;
	const part1HasBrackets = leftTargetSubStr.length > 1;
	const part2HasBrackets = rightTargetSubStr.length > 1;
	if (part1HasBrackets) { lengthOfOriginalTokens += 2; }
	if (part2HasBrackets) { lengthOfOriginalTokens += 2; }

	for (let wordIndex = 0; wordIndex < words.length; wordIndex++) {
		const currentWord = words[wordIndex];
		if (currentWord.length < lengthOfOriginalTokens) {
			continue;
		}
		// NEW!
		if (currentWord[0] === '{' && currentWord.indexOf('}') === (currentWord.length - 1)) {
			continue;
		}

		for (let subWordIndex = 0; subWordIndex < currentWord.length - 1; subWordIndex++) {
			if (subWordIndex + lengthOfOriginalTokens > currentWord.length) {
				break;	
			}
			const originalSubWordIndex = subWordIndex;

			let part1 = part1HasBrackets ? '' : currentWord[subWordIndex];
			if (!part1HasBrackets && currentWord[subWordIndex] === '{') {
				for (; subWordIndex < currentWord.length; subWordIndex++) {
					if (currentWord[subWordIndex] === '}') {
						break;
					}
				}
				continue;
			}
			if (part1HasBrackets && currentWord[subWordIndex] === '{') {
				let iteratorIndexRelativeToWord = 0;
				let iterator = subWordIndex + 1;
				
				for (; iterator < currentWord.length; iterator++) {
					if (currentWord[iterator] === '}') {
						break;
					}

					part1 += currentWord[iterator];
					iteratorIndexRelativeToWord++;
				}

				subWordIndex = iterator;
			}
			if (part1 !== leftTargetSubStr) {
				continue;
			}		

			let part2 = part2HasBrackets ? '' : currentWord[subWordIndex + 1];
			if (part2HasBrackets && currentWord[subWordIndex + 1] === '{') {
				let iteratorIndexRelativeToWord = 0;
				let iterator = subWordIndex + 2;
				for (; iterator < currentWord.length; iterator++) {
					if (currentWord[iterator] === '}') {
						break;
					}

					part2 += currentWord[iterator];
					iteratorIndexRelativeToWord++;
				}
			}
			if (part2 !== rightTargetSubStr) {
				// if part2 has no brackets, and part2 will be a single bracket ({), this condition will always trigger) 
				continue;
			}

			let wordPostMerge = '';
			for (let i = 0; i < currentWord.length; i++) {
				if (i < originalSubWordIndex) {
					wordPostMerge += currentWord[i];
				}
				// 0, 1, 2 (index 2), { (index 3), 3 (index 4), 4 (index 5), 5 (index 6), } (index 7), 6 {index 8}
				else if (i >= originalSubWordIndex + lengthOfOriginalTokens) {
					wordPostMerge += currentWord[i];
				} else {
					wordPostMerge += '{';
					for (let mergedWordIterator = 0; mergedWordIterator < finalMergedWord.length; mergedWordIterator++) {
						wordPostMerge += finalMergedWord[mergedWordIterator];
					}
					wordPostMerge += '}';
					i = i + lengthOfOriginalTokens - 1; // -1 because loop will increment i at the next iteration
				}
			}

			words[wordIndex] = wordPostMerge;
			wordIndex--; // check same word again for another repition of the same combo
			break; // need to break because the length of the currentWord changed with the merge
		}
	}

	return words;
}

globalThis.flattenFinalTokenizedWordsArr = (wordsArr) => {
	const flattenedTokensArr = [];
	for (let wordIndex = 0; wordIndex < wordsArr.length; wordIndex++) {
		const currentWord = wordsArr[wordIndex];
		for (let subWordIndex = 0; subWordIndex < currentWord.length; subWordIndex++) {
			if (currentWord[subWordIndex] === '{') {
				let mergedToken = '';
				subWordIndex++;
				for (; subWordIndex < currentWord.length; subWordIndex++) {
					if (currentWord[subWordIndex] === '}') {
						flattenedTokensArr.push(mergedToken);
						break;
					} else {
						mergedToken += currentWord[subWordIndex];
					}
				}	
			} else {
				flattenedTokensArr.push(currentWord[subWordIndex]);
			}
		}
	}
	return flattenedTokensArr;
}