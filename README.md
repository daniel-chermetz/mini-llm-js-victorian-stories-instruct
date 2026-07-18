# A story-teller LLM in the style of 19th century Victorian Britain
# https://daniel-chermetz.github.io/mini-llm-js-victorian-stories-instruct/

Small language model trained on short stories in the genres of romance and Gothic suspence, premised on magical realism, and all in the style of 19th century Victorian Britain.

Offering a no-download one click online browser based inference through WebGPU at:
https://daniel-chermetz.github.io/mini-llm-js-victorian-stories-instruct/

It would take two minutes or so for the model to stream to the browser's memory before it can be used. Alternatively, the repo can be downloaded, and served through localhost (npx serve), making the model load almost instantenously. (It's a pretty simple process, and AI can explain the requisite technical steps, with these being straightforward and of very common usage.)

Originally I wrote most parts of the project, but AI has been delegated a greater role over time (for which help I'm grateful).

Currently, I write line-by-line, 100% by hand, every single kernel / shader in the inference workflow, and additionally the core BPE API - divide text into words, BPE tokenize words by a custom vocab, and BPE train custom vocab on a body of text (only needs to be performed once as the first step before training a new model).

(On the other, AI is now fully responsible for orchestrating the entire workflow by saving, staging and moving data throughout the inference workflow from one kernel to the next, as well as from host to device and back again, while lunching kernels in the correct order and with the requisite inputs; in short, AI is responsible for all operations outside the kernels / shaders themselves. Outside of the inference process, AI is responsible for entierty of the UI implementation, and for calling my BPE functions to tokenize context text in preparation for inference.)

(In the training project, also on Github, I also manually write by hand all infrence, training and optimizer kernels & cuBLAS calls, while leaving everything else to AI.)
