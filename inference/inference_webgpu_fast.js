globalThis.Transformer = function(tIndex, L_param) {
	const dimensions = NetworkMeta.dimensions;
	const headDim = dimensions / NetworkMeta.heads;
	let L = L_param;

	this.RMSNorm = (xInputs, rmsGamma, rmsSumBuffer, rmsOutputBuffer) => {
		const xInputsPostRMSGamma = lmNetwork.RMSNorm_WEBGPU(xInputs, rmsGamma, rmsSumBuffer, rmsOutputBuffer);
		return xInputsPostRMSGamma;
	}

	this.generateVals = () => {
		this.vals = lmNetwork.matMul_dim_dim_WEBGPU(this.valueWeights, this.inputsPostRMS, lmNetwork.preBuffersByTransformer[tIndex].matMulV);
		this.valsByHead = this.vals;
	}

	this.generateKeys = () => {
		this.keys = lmNetwork.matMul_dim_dim_WEBGPU(this.keyWeights, this.inputsPostRMS, lmNetwork.preBuffersByTransformer[tIndex].matMulK);
		this.keysByHead = this.keys;
	}

	this.generateQueries = () => {
		this.queries = lmNetwork.matMul_dim_dim_WEBGPU(this.queryWeights, this.inputsPostRMS, lmNetwork.preBuffersByTransformer[tIndex].matMulQ);
		this.queriesByHead = this.queries;
	}

	if (NetworkMeta.CONFIG_QK_RMS) {
		this.QKRMSNormByHead = () => {
		    const pre = lmNetwork.preBuffersByTransformer[tIndex];
		    // 1) denominators per head/column
		    lmNetwork.colSumRMSNormByHead_WEBGPU(this.keysByHead,    pre.qkRmsK);
		    lmNetwork.colSumRMSNormByHead_WEBGPU(this.queriesByHead, pre.qkRmsQ);
		    // 2) normalize + gamma into fresh buffers, and re-point keysByHead/queriesByHead
		    this.keysByHead    = lmNetwork.RMSNormByHead_WEBGPU(this.keysByHead,    pre.qkRmsK, this.rmsGammaKeys,    pre.qkNormedK);
		    this.queriesByHead = lmNetwork.RMSNormByHead_WEBGPU(this.queriesByHead, pre.qkRmsQ, this.rmsGammaQueries, pre.qkNormedQ);
		};
	}

	this.applyRoPE = (keysOrQueriesByHead, ropeOutputBuffer) => {
		if (!lmNetwork.precomputedTheta) {
			const flatPrecomputedTheta = new Float32Array(headDim * L);

			const headPairs = headDim / 2;
			for (let colIndex = 0; colIndex < L; colIndex++) {
				for (let pairIndex = 0; pairIndex < headPairs; pairIndex++) {
					flatPrecomputedTheta[colIndex * headDim + pairIndex * 2] = Math.cos(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim)));
					flatPrecomputedTheta[colIndex * headDim + pairIndex * 2 + 1] = (Math.sin(colIndex / Math.pow(NetworkMeta.ropeDenomBase, (2 * pairIndex / headDim))));
				}
			}
			
			lmNetwork.precomputedTheta = globalThis.prepareFastPrecomputedTheta(lmNetwork, flatPrecomputedTheta, headDim, L);
		}

		const keysOrQueriesByHeadPostRoPE = lmNetwork.RoPE_WEBGPU(keysOrQueriesByHead, lmNetwork.precomputedTheta, ropeOutputBuffer);
		return keysOrQueriesByHeadPostRoPE;
	}

	this.getAttentionScoresByHead = () => {	
		const attentionByHeadPreSoftmaxScaledMasked = lmNetwork.matMul_KtQ_WEBGPU(tIndex);
		this.attentionByHeadPostSoftmax = lmNetwork.softmaxByHead_WEBGPU(tIndex);
		this.valueScaledAttentionByHead = lmNetwork.matMulValsAttention_WEBGPU(tIndex);
	}

	if (NetworkMeta.CONFIG_QUERY_GATING) {
		this.generateGatedQueries = () => {
			this.gatedQueries = lmNetwork.matMul_dim_dim_WEBGPU(this.queryGateWeights, this.inputsPostRMS, lmNetwork.preBuffersByTransformer[tIndex].gatedQueries);
		}
		this.applySigmoidToGatedQueries = () => {
			this.gatedQueriesPostSigmoid = lmNetwork.applySigmoid_WEBGPU(this.gatedQueries, tIndex);
		}
		this.applyQueryGatingHadamard = () => {
			this.queryGatedValueScaledAttnByHead = lmNetwork.applyQueryGatingHadamard_WEBGPU(this.valueScaledAttentionByHead, this.gatedQueriesPostSigmoid, tIndex);
		}
	}

	this.projectConcatAttentionToOutput = () => {
		this.outputProjectedAttention = lmNetwork.matMul_dim_dim_WEBGPU(this.outputProjectionWeights, NetworkMeta.CONFIG_QUERY_GATING ? this.queryGatedValueScaledAttnByHead : this.valueScaledAttentionByHead, lmNetwork.preBuffersByTransformer[tIndex].outputProj);
	}

	this.addResidualInputsToConcatAttentionOutput = () => {
		this.outputProjectedAttentionScoresPlusResidualInputs = lmNetwork.elementWiseAdd_WEBGPU(this.outputProjectedAttention, this.inputs, lmNetwork.preBuffers.residual1);
	}

	this.applyFeedforwardToAttention = () => {
		const left1MatrixPreSilu = lmNetwork.matMul_FFN1_WEBGPU(this.feedForwardWeights1A, this.outputProjectedAttentionScoresPostRMS2, lmNetwork.preBuffersByTransformer[tIndex].ffn1a);
		const left1Matrix = lmNetwork.applySilu_WEBGPU(left1MatrixPreSilu, tIndex);
		const right1Matrix = lmNetwork.matMul_FFN1_WEBGPU(this.feedForwardWeights1B, this.outputProjectedAttentionScoresPostRMS2, lmNetwork.preBuffersByTransformer[tIndex].ffn1b);

		const final1Matrix = lmNetwork.applyHadamard_WEBGPU(left1Matrix, right1Matrix, tIndex);
		
		this.attentionPostFeedForward = lmNetwork.matMul_FFN2_WEBGPU(this.feedForwardWeights2, final1Matrix, tIndex);
	}

	this.getFinalTransformerOutputByAddingResidualToFFNResult = () => {
		this.finalTransformerOutput = lmNetwork.elementWiseAdd_WEBGPU(this.attentionPostFeedForward, this.outputProjectedAttentionScoresPlusResidualInputs, lmNetwork.preBuffers.residual2);
	}

	this.applyTransformChainToInputs = (downstreamInputs) => {
		this.inputs = downstreamInputs;
		this.inputsPostRMS = this.RMSNorm(this.inputs, this.rmsGamma, lmNetwork.preBuffersByTransformer[tIndex].rms1Sum, lmNetwork.preBuffersByTransformer[tIndex].rms1);

		this.generateVals();
		this.generateKeys();
		this.generateQueries();

		if (NetworkMeta.CONFIG_QK_RMS) {
			this.QKRMSNormByHead();
		}

		this.keysByHeadPostRoPE = this.applyRoPE(this.keysByHead, lmNetwork.preBuffersByTransformer[tIndex].ropeK);
		this.queriesByHeadPostRoPE = this.applyRoPE(this.queriesByHead, lmNetwork.preBuffersByTransformer[tIndex].ropeQ);
		lmNetwork.setNewValuesKeys_WEBGPU(this.valsByHead, this.keysByHeadPostRoPE, tIndex);

		this.getAttentionScoresByHead();

		if (NetworkMeta.CONFIG_QUERY_GATING) {
			this.generateGatedQueries();
			this.applySigmoidToGatedQueries();
			this.applyQueryGatingHadamard();
		}

		this.projectConcatAttentionToOutput();
		this.addResidualInputsToConcatAttentionOutput();

		this.outputProjectedAttentionScoresPostRMS2 = this.RMSNorm(this.outputProjectedAttentionScoresPlusResidualInputs, this.rmsGamma2, lmNetwork.preBuffersByTransformer[tIndex].rms2Sum, lmNetwork.preBuffersByTransformer[tIndex].rms2);
		
		this.applyFeedforwardToAttention();
		this.getFinalTransformerOutputByAddingResidualToFFNResult();

		return this.finalTransformerOutput;
	}
}

globalThis.LmNetwork = function () {
  	this.init = async () => {
    	const adapter = await navigator.gpu.requestAdapter();
    	if (!adapter) throw new Error("WebGPU adapter not available");
    	this.webgpuDevice = await adapter.requestDevice();
  	}
	
	this.setShaders = () => {	
		if (!this.load_dim_dim_weights_WEBGPU) {
			const dimDimShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let dimensions = ${dimensions}u;
					
					if (col < dimensions && row < dimensions) {
						let index = row * dimensions + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_dim_dim_weights_WEBGPU = function(weightMatrix) {			
				const flatInput = new Float32Array(dimensions * dimensions);
				for (let row = 0; row < dimensions; row++) {
					for (let col = 0; col < dimensions; col++) {
						flatInput[row * dimensions + col] = weightMatrix[row][col];
					}
				}

				return executeDimDimWeightLoading(this, flatInput, dimensions, dimDimShaderCode);
			};
		}

		if (!this.load_rms_weights_WEBGPU) {
			const rmsShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(64)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let index = global_id.x;
					let dimensions = ${dimensions}u;
					
					if (index < dimensions) {
						output[index] = input[index];
					}
				}
			`;

			this.load_rms_weights_WEBGPU = function(weightMatrix) {
				const flatInput = new Float32Array(weightMatrix);		
				return executeRmsWeightLoading(this, flatInput, dimensions, rmsShaderCode);
			};
		}
		
		if (!this.load_ffn1_weights_WEBGPU) {
			const ffn1ShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${dimensions}u;
					let numRows = ${NetworkMeta.ffnDim}u;
					
					if (col < numCols && row < numRows) {
						let index = row * numCols + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_ffn1_weights_WEBGPU = function(weightMatrix) {
				const numRows = NetworkMeta.ffnDim;
				const numCols = dimensions;
				
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeFfn1WeightLoading(this, flatInput, numRows, numCols, ffn1ShaderCode);
			};
		}
		
		if (!this.load_ffn2_weights_WEBGPU) {
			const ffn2ShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${NetworkMeta.ffnDim}u;
					let numRows = ${dimensions}u;
					
					if (col < numCols && row < numRows) {
						let index = row * numCols + col;
						output[index] = input[index];
					}
				}
			`;

			this.load_ffn2_weights_WEBGPU = function(weightMatrix) {
				const numRows = dimensions;
				const numCols = NetworkMeta.ffnDim;
				
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeFfn2WeightLoading(this, flatInput, numRows, numCols, ffn2ShaderCode);
			};
		}

		if (!this.load_token_embeddings_WEBGPU) {
			const numRows = vocabSize;
			const numCols = dimensions;

			const tokenEmbeddingsShaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(8, 8)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let col = global_id.x;
					let row = global_id.y;
					let numCols = ${numCols}u;
					let numRows = ${numRows}u;
					
					if (col < numCols && row < numRows) {
						let inputIndex = row * numCols + col;
						let outputIndex = row * numCols + col;
						output[outputIndex] = input[inputIndex];
					}
				}
			`;

			this.load_token_embeddings_WEBGPU = function(weightMatrix) {			
				const flatInput = new Float32Array(numRows * numCols);
				for (let row = 0; row < numRows; row++) {
					for (let col = 0; col < numCols; col++) {
						flatInput[row * numCols + col] = weightMatrix[row][col];
					}
				}

				return executeTokenEmbeddingsWeightLoading(this, flatInput, numRows, numCols, tokenEmbeddingsShaderCode);
			};
		}

		if (!this.setInputTokenEmbeddings_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> indices: array<u32>;
				@group(0) @binding(1) var<storage, read> tokenEmbeddings: array<f32>;
				@group(0) @binding(2) var<storage, read> rightEndIndexArr: array<u32>;
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {					
					const dimensions = ${dimensions}u;
					let row = global_id.x; // dimensions

					if (row >= dimensions) {
						return;
					}

					let current_L_index = rightEndIndexArr[0];
					let tokenIndex = indices[current_L_index];
					let embeddingIndex = tokenIndex * dimensions + row;
					output[row] = tokenEmbeddings[embeddingIndex];
				}
			`;

			this.setInputTokenEmbeddings_WEBGPU = function(tokenEmbeddings) {
				return executeFastSetInputTokenEmbeddings(this, tokenEmbeddings.buffer, dimensions, shaderCode);
			};
		}

		if (!this.RMSNorm_WEBGPU) {
			const shaderCode_RMS_sum = `
				@group(0) @binding(0) var<storage, read> xInputs: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				var<workgroup> sumArr: array<f32, 64>;

				@compute @workgroup_size(64)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let tId = global_id.x;

					var sum = 0.0f;
					for (var i = tId; i < ${dimensions}u; i+= 64u) {
						sum += (xInputs[i] * xInputs[i]);
					}

					sumArr[tId] = sum;
					workgroupBarrier(); 

					for (var reductionSize = 32u; reductionSize > 0u; reductionSize /= 2u) {
						if (tId < reductionSize) {
							sumArr[tId] = (sumArr[tId] + sumArr[tId + reductionSize]);
						}

						workgroupBarrier();
					}

					if (tId == 0u) {
						let denominator = sqrt((sumArr[0] / f32(${dimensions})) + 1e-8);
						output[0] = denominator;	
					}
				}
			`;

			const shaderCode_RMS_by_coord = `
				@group(0) @binding(0) var<storage, read> xInputs: array<f32>;
				@group(0) @binding(1) var<storage, read> rmsSum: array<f32>;
				@group(0) @binding(2) var<storage, read> rmsGamma: array<f32>;				
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;
					if (row >= ${dimensions}u) {
						return;
					}
					
					let normalized = xInputs[row] / rmsSum[0];
					output[row] = normalized * rmsGamma[row];
				}
			`;			

			this.RMSNorm_WEBGPU = function(xInputs, rmsGamma, rmsSumBuffer, rmsOutputBuffer) {
				return executeFastRMSNorm(this, xInputs.buffer, rmsGamma.buffer, dimensions, shaderCode_RMS_sum, shaderCode_RMS_by_coord, rmsSumBuffer, rmsOutputBuffer);
			};
		}

		if (NetworkMeta.CONFIG_QK_RMS && !this.colSumRMSNormByHead_WEBGPU) {
			const shaderCode_QK_RMS_sum = `
				@group(0) @binding(0) var<storage, read> xInputs: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;
				
				var<workgroup> sumArr: array<f32, 32>;

				@compute @workgroup_size(32)
				fn main(
					@builtin(workgroup_id) workgroupId: vec3<u32>,
					@builtin(local_invocation_index) tId: u32
				) {
					let headIndex = workgroupId.x;
					let headDim = ${headDim}u;
					let headOffset = headIndex * headDim;

					var sum = 0.0f;
					for (var i = headOffset + tId; i < headOffset + headDim; i+= 32u) {
						sum += (xInputs[i] * xInputs[i]);
					}

					sumArr[tId] = sum;
					workgroupBarrier(); 

					for (var reductionSize = 16u; reductionSize > 0u; reductionSize /= 2u) {
						if (tId < reductionSize) {
							sumArr[tId] = (sumArr[tId] + sumArr[tId + reductionSize]);
						}
						workgroupBarrier();
					}

					if (tId == 0u) {
						let denominator = sqrt((sumArr[0] / f32(${headDim})) + 1e-8);
						output[headIndex] = denominator;	
					}
				}
			`;

			const shaderCode_QK_RMS_by_coord = `
				@group(0) @binding(0) var<storage, read> xInputs: array<f32>;
				@group(0) @binding(1) var<storage, read> rmsSumByHead: array<f32>;
				@group(0) @binding(2) var<storage, read> rmsGamma: array<f32>;				
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;
					if (row >= ${dimensions}u) {
						return;
					}
					
					let headIndex = row / ${headDim}u;
					let normalized = xInputs[row] / rmsSumByHead[headIndex];					
					output[row] = normalized * rmsGamma[row];
				}				
			`

			this.colSumRMSNormByHead_WEBGPU = function(xInputs, colSumRMSByHeadOutputBuffer) {
				return executeFastColSumRMSNormByHead(this, xInputs.buffer, heads, shaderCode_QK_RMS_sum, colSumRMSByHeadOutputBuffer);
			};
			this.RMSNormByHead_WEBGPU = function(xInputs, denominatorBuffer, rmsGamma, outputBuffer) {
				return executeFastRMSNormByHead(this, xInputs.buffer, denominatorBuffer, rmsGamma.buffer, dimensions, shaderCode_QK_RMS_by_coord, outputBuffer);
			};
		}

		if (!this.matMul_dim_dim_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> a: array<f32>;
				@group(0) @binding(1) var<storage, read> b: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x; // dimensions				
					let dimensions = ${dimensions}u;
					if (row >= dimensions) {
						return;
					}
					
					let rowOffset = row * dimensions;

					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
						let aIndex = rowOffset + i;
						let bIndex = i;
						sum = sum + (a[aIndex] * b[bIndex]);
					}
					output[row] = sum;
				}
			`;

			this.matMul_dim_dim_WEBGPU = function(a, b, outputBuffer) {
				return executeFastMatMulDimDim(this, a.buffer, b.buffer, dimensions, shaderCode, outputBuffer);
			};
		}

		if (!this.RoPE_WEBGPU) {
			const headPairs = headDim / 2;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read> theta: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(2, 32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {				
					let headIndex = global_id.x;
					let pair = global_id.y; // headPairs (headDim / 2)

					let heads = ${heads}u;
					let headPairs = ${headPairs}u;

					if (headIndex >= heads || pair >= headPairs) {
						return;
					}

					let headDim = ${headDim}u;
					let headOffset = headIndex * headDim;

					let headRelativeRow = pair * 2u; // always even!
					let globalRow = headOffset + headRelativeRow;

					// theta lookup: headRelativeRow + (0 for cos, 1 for sin)
					let cosVal = theta[headRelativeRow];
					let sinVal = theta[headRelativeRow + 1u];

					let x: f32 = input[globalRow];
					let y: f32 = input[globalRow + 1u];

					// even: cos * x - sin * y
					output[globalRow] = cosVal * x - sinVal * y;
					// odd: sin * x + cos * y
					output[globalRow + 1u] = sinVal * x + cosVal * y;
				}
			`;

			this.RoPE_WEBGPU = function(keysOrQueriesByHead, precomputedTheta, ropeOutputBuffer) {
				return executeFastRoPE(this, keysOrQueriesByHead.buffer, precomputedTheta, headDim, heads, shaderCode, ropeOutputBuffer);
			};
		}

		if (!this.matMul_KtQ_WEBGPU) {
			const sqrtHeadDim = Math.sqrt(headDim);

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> keys: array<f32>;
				@group(0) @binding(1) var<storage, read> queries: array<f32>;
				@group(0) @binding(2) var<storage, read> rightEndIndexArr: array<u32>;
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(2, 32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let headIndex = global_id.x;
					let outRow = global_id.y;

					let heads = ${heads}u;
					let L = ${L}u;

					if (headIndex >= heads || outRow >= L) {
						return;
					}

					let outputHeadOffset = headIndex * L;
					let outputIndex = outputHeadOffset + outRow;

					if (outRow > rightEndIndexArr[0]) {
						output[outputIndex] = 0.0f; // mask
						return;
					}

					let headDim = ${headDim}u;
					let sqrtHeadDim = ${sqrtHeadDim}f;

					let inputHeadOffset = headIndex * headDim;
					let kHeadOffset = inputHeadOffset * L;
					let kHeadRowOffset = kHeadOffset + outRow;

					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < headDim; i = i + 1u) {
						sum = sum + keys[kHeadRowOffset + i * L] * queries[inputHeadOffset + i];
					}
					output[outputIndex] = sum / sqrtHeadDim;
				}
			`;

			this.matMul_KtQ_WEBGPU = function(tIndex) {
				return executeFastKtQ(this, heads, L, shaderCode, tIndex);
			};
		}

		if (!this.softmaxByHead_WEBGPU) {
			const col_max_sum_softmax_shader = `
			@group(0) @binding(0) var<storage, read> attnScores: array<f32>;
			@group(0) @binding(1) var<storage, read> rightEndIndexArr: array<u32>;
			@group(0) @binding(2) var<storage, read_write> maxOutput: array<f32>;
			@group(0) @binding(3) var<storage, read_write> sumOutput: array<f32>;				
			
			var<workgroup> maxArr: array<f32, 32>;
			var<workgroup> sumArr: array<f32, 32>;

			@compute @workgroup_size(32)
			fn main(
				@builtin(workgroup_id) workgroupId: vec3<u32>,
				@builtin(local_invocation_index) tId: u32
			) {
				let headIndex = workgroupId.x;
				let L = ${L}u;
				let headOffset = headIndex * L;
				let current_L_index = rightEndIndexArr[0];

				var localMax: f32 = -3.402823e+38f;
				for (var i = tId; i <= current_L_index; i = i + 32u) {
					let attnVal = attnScores[headOffset + i];
					if (attnVal > localMax) {
						localMax = attnVal;
					}
				}
				maxArr[tId] = localMax;
				workgroupBarrier(); 


				for (var reductionSize = 16u; reductionSize > 0u; reductionSize /= 2u) {
					if (tId < reductionSize) {
						if (maxArr[tId + reductionSize] > maxArr[tId]) {
							maxArr[tId] = maxArr[tId + reductionSize];
						}
					}
					workgroupBarrier();
				}

				let headMax = maxArr[0];

				if (tId == 0u) {
					maxOutput[headIndex] = headMax;
				}

				var localSum: f32 = 0.0f;
				for (var i = tId; i <= current_L_index; i = i + 32u) {
					localSum += exp(attnScores[headOffset + i] - headMax);
				}
				sumArr[tId] = localSum;
				workgroupBarrier();

				for (var reductionSize = 16u; reductionSize > 0u; reductionSize /= 2u) {
					if (tId < reductionSize) {
						sumArr[tId] = (sumArr[tId] + sumArr[tId + reductionSize]);
					}
					workgroupBarrier();
				}

				if (tId == 0u) {
					sumOutput[headIndex] = sumArr[0];
				}
			}
			`;

			const col_per_coord_softmax_shader = `
			@group(0) @binding(0) var<storage, read> attnScores: array<f32>;
			@group(0) @binding(1) var<storage, read> maxByCol: array<f32>;
			@group(0) @binding(2) var<storage, read> sumByCol: array<f32>;
			@group(0) @binding(3) var<storage, read> rightEndIndexArr: array<u32>;
			@group(0) @binding(4) var<storage, read_write> softmaxAttnScores: array<f32>;

			@compute @workgroup_size(32)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				let rowIndex = global_id.x;
				let L = ${L}u;
				if (rowIndex >= (${heads}u * L)) {
					return;
				}

				let headIndex = rowIndex / L;
				let headOffset = headIndex * L;
				let headRelativeRow = rowIndex - headOffset;

				if (headRelativeRow > rightEndIndexArr[0]) {
					softmaxAttnScores[rowIndex] = 0.0f; // masking
					return;
				}

				softmaxAttnScores[rowIndex] = exp(attnScores[rowIndex] - maxByCol[headIndex]) / sumByCol[headIndex];
			}
			`;

			this.softmaxByHead_WEBGPU = function(tIndex) {
				return executeFastSoftmaxByHead(this, heads, L, col_max_sum_softmax_shader, col_per_coord_softmax_shader, tIndex);
			}
		}

		if (!this.matMulValsAttention_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> vals: array<f32>;
				@group(0) @binding(1) var<storage, read> attention: array<f32>;
				@group(0) @binding(2) var<storage, read> rightEndIndexArr: array<u32>;
				@group(0) @binding(3) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(2, 32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let headIndex = global_id.x;
					let outRow = global_id.y;

					let heads = ${heads}u;
					let headDim = ${headDim}u;
					let L = ${L}u;

					if (headIndex >= heads || outRow >= headDim) {
						return;
					}

					let attentionHeadOffset = headIndex * L;
					let valuesHeadRowOffset = (headDim * headIndex + outRow) * L;

					let outputIndex = headIndex * headDim + outRow;
					let current_L_index = rightEndIndexArr[0];

					var sum: f32 = 0.0f;
					for (var i: u32 = 0u; i <= current_L_index; i = i + 1u) {
						sum = sum + vals[valuesHeadRowOffset + i] * attention[attentionHeadOffset + i];
					}
					output[outputIndex] = sum;
				}
			`;

			this.matMulValsAttention_WEBGPU = function(tIndex) {
				return executeFastMatMulValsAttention(this, headDim, heads, shaderCode, tIndex);
			};
		}

		if (NetworkMeta.CONFIG_QUERY_GATING && !this.applySigmoid_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x; // dimensions
					if (row >= ${dimensions}u) {
						return;
					}

					let x = input[row];
					// Sigmoid: 1 / (1 + exp(-x))
					output[row] = 1.0 / (1.0 + exp(-x));
				}
			`;

			this.applySigmoid_WEBGPU = function(input, tIndex) {
				return executeFastSigmoid(this, input.buffer, dimensions, shaderCode, tIndex);
			};
		}

		if (!this.applyQueryGatingHadamard_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> left: array<f32>;
				@group(0) @binding(1) var<storage, read> right: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x; // dimensions
					if (row >= ${dimensions}u) {
						return;
					}

					output[row] = left[row] * right[row];
				}
			`;

			this.applyQueryGatingHadamard_WEBGPU = function(left, right, tIndex) {
				return executeFastQueryGatingHadamard(this, left.buffer, right.buffer, dimensions, shaderCode, tIndex);
			};
		}

		if (!this.elementWiseAdd_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> a: array<f32>;
				@group(0) @binding(1) var<storage, read> b: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x; // dimensions
					if (row >= ${dimensions}u) {
						return;
					}
					
					output[row] = a[row] + b[row];
				}
			`;

			this.elementWiseAdd_WEBGPU = function(a, b, outputBuffer) {
				return executeFastElementWiseAdd(this, a.buffer, b.buffer, dimensions, shaderCode, outputBuffer);
			};
		}

		if (!this.matMul_FFN1_WEBGPU) {
			const ffnDim = NetworkMeta.ffnDim;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> weights: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x; // dimensions
					if (row >= ${ffnDim}u) {
						return;
					}
					let dimensions = ${dimensions}u;
					let weightRowOffset = row * dimensions;

					var sum: f32 = 0.0f;
					for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
						sum = sum + weights[weightRowOffset + i] * input[i];
					}
					output[row] = sum;
				}
			`;

			this.matMul_FFN1_WEBGPU = function(weights, input, ffnOutputBuffer) {
				return executeFastMatMulFFN1(this, weights.buffer, input.buffer, ffnDim, shaderCode, ffnOutputBuffer);
			};
		}
		
		if (!this.applySilu_WEBGPU) {
			const ffnDim = NetworkMeta.ffnDim;
			
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> input: array<f32>;
				@group(0) @binding(1) var<storage, read_write> output: array<f32>;

				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;
					if (row >= ${ffnDim}u) {
						return;
					}
					
					let x = input[row];
					// SiLU: x / (1 + exp(-x))
					output[row] = x / (1.0 + exp(-x));
				}
			`;

			this.applySilu_WEBGPU = function(input, tIndex) {
				return executeFastSilu(this, input.buffer, ffnDim, shaderCode, tIndex);
			};
		}

		if (!this.applyHadamard_WEBGPU) {
			const ffnDim = NetworkMeta.ffnDim;
			
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> left: array<f32>;
				@group(0) @binding(1) var<storage, read> right: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;
					if (row >= ${ffnDim}u) {
						return;
					}

					output[row] = left[row] * right[row];
				}
			`;

			this.applyHadamard_WEBGPU = function(left, right, tIndex) {
				return executeFastHadamard(this, left.buffer, right.buffer, ffnDim, shaderCode, tIndex);
			};
		}

		if (!this.matMul_FFN2_WEBGPU) {
			const ffnDim = NetworkMeta.ffnDim;

			const shaderCode = `
				@group(0) @binding(0) var<storage, read> weights: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(4) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;					
					if (row >= ${dimensions}u) {
						return;
					}
					let ffnDim = ${ffnDim}u;
					let weightRowOffset = ffnDim * row;

					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < ffnDim; i = i + 1u) {
						sum = sum + weights[weightRowOffset + i] * input[i];
					}					
					output[row] = sum;
				}
			`;

			this.matMul_FFN2_WEBGPU = function(weights, input, tIndex) {
				return executeFastMatMulFFN2(this, weights.buffer, input.buffer, dimensions, shaderCode, tIndex);
			}
		}

		if (!this.matMul_vocab_WEBGPU) {
			const shaderCode = `
				@group(0) @binding(0) var<storage, read> embeddings: array<f32>;
				@group(0) @binding(1) var<storage, read> input: array<f32>;
				@group(0) @binding(2) var<storage, read_write> output: array<f32>;
				
				@compute @workgroup_size(32)
				fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
					let row = global_id.x;					
					if (row >= ${vocabSize}u) {
						return;
					}
					let dimensions = ${dimensions}u;
					let vocabRowOffset = dimensions * row;

					var sum: f32 = 0.0;
					for (var i: u32 = 0u; i < dimensions; i = i + 1u) {
						sum = sum + embeddings[vocabRowOffset + i] * input[i];
					}
					output[row] = sum;
				}
			`;

			this.matMul_vocab_WEBGPU = function(embeddings, input) {		
				return executeFastMatMulVocab(this, embeddings.buffer, input.buffer, vocabSize, shaderCode);
			};
		}

		// new shader find max among vocab scores
		const vocab_scores_max_shader = `
			@group(0) @binding(0) var<storage, read> vocabScores: array<f32>;
			@group(0) @binding(1) var<storage, read> rightEndIndexArr: array<u32>;
			@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
			
			var<workgroup> maxArr: array<f32, 128>;
			var<workgroup> maxIndexArr: array<u32, 128>;

			@compute @workgroup_size(128)
			fn main(
				@builtin(workgroup_id) workgroupId: vec3<u32>,
				@builtin(local_invocation_index) tId: u32
			) {
				var localMax: f32 = -3.402823e+38f;
				var localMaxIndex: u32 = 0;
				for (var i = tId; i < ${vocabSize}u; i = i + 128u) {
					let attnVal = vocabScores[i];
					if (attnVal > localMax) {
						localMax = attnVal;
						localMaxIndex = i;
					}
				}
				maxArr[tId] = localMax;
				maxIndexArr[tId] = localMaxIndex;
				workgroupBarrier(); 

				for (var reductionSize = 64u; reductionSize > 0u; reductionSize /= 2u) {
					if (tId < reductionSize) {
						if (maxArr[tId + reductionSize] > maxArr[tId]) {
							maxArr[tId] = maxArr[tId + reductionSize];
							maxIndexArr[tId] = maxIndexArr[tId + reductionSize]; 
						}
					}
					workgroupBarrier();
				}

				if (tId == 0u) {
					indices[rightEndIndexArr[0] + 1u] = maxIndexArr[0];
				}
			}
		`;

		const set_new_values_keys_shader = `
			@group(0) @binding(0) var<storage, read> newTokenValues: array<f32>;
			@group(0) @binding(1) var<storage, read> newTokenKeys: array<f32>;
			@group(0) @binding(2) var<storage, read> rightEndIndexArr: array<u32>;
			@group(0) @binding(3) var<storage, read_write> values: array<f32>;
			@group(0) @binding(4) var<storage, read_write> keys: array<f32>;
			
		
			@compute @workgroup_size(32)
			fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
				let row = global_id.x;	
				if (row >= ${dimensions}u) {
					return;
				}

				let index = rightEndIndexArr[0] + 1u + ${L}u * row;
				values[index] = newTokenValues[row];
				keys[index] = newTokenKeys[row];
		}			
		`

		this.selectNextToken_WEBGPU = function(vocabScores) {
			return executeFastVocabMax(this, vocabScores.buffer, vocab_scores_max_shader);
		};

		this.setNewValuesKeys_WEBGPU = function(newTokenValues, newTokenKeys, tIndex) {
			return executeFastSetNewValuesKeys(this, newTokenValues.buffer, newTokenKeys.buffer, dimensions, set_new_values_keys_shader, tIndex);
		};
	}

	this.setTransformers = () => {	
		this.transformers = [];
		for (let tIndex = 0; tIndex < NetworkMeta.numTransformers; tIndex++) {
			this.transformers.push(new Transformer(tIndex, L));
			
			this.transformers[this.transformers.length - 1].queryWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].queryWeights);
			if (NetworkMeta.CONFIG_QUERY_GATING) {
				this.transformers[this.transformers.length - 1].queryGateWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].queryGateWeights);
			}

			this.transformers[this.transformers.length - 1].keyWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].keyWeights);
			this.transformers[this.transformers.length - 1].valueWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].valueWeights);
			
			this.transformers[this.transformers.length - 1].rmsGamma = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGamma);
			this.transformers[this.transformers.length - 1].rmsGamma2 = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGamma2);

			if (NetworkMeta.CONFIG_QK_RMS) {
				this.transformers[this.transformers.length - 1].rmsGammaKeys = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGammaKeys);
				this.transformers[this.transformers.length - 1].rmsGammaQueries = this.load_rms_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].rmsGammaQueries);
			}
			
			this.transformers[this.transformers.length - 1].outputProjectionWeights = this.load_dim_dim_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].outputProjectionWeights);
			
			this.transformers[this.transformers.length - 1].feedForwardWeights1A = this.load_ffn1_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights1A);
			this.transformers[this.transformers.length - 1].feedForwardWeights1B = this.load_ffn1_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights1B);
			
			this.transformers[this.transformers.length - 1].feedForwardWeights2 = this.load_ffn2_weights_WEBGPU(GLOBAL_WEIGHTS[tIndex].feedForwardWeights2);
		}
	}

	const dimensions = NetworkMeta.dimensions;
	const heads = NetworkMeta.heads;
	const headDim = NetworkMeta.dimensions / NetworkMeta.heads;
	let L = 256;
	this.tokenVals = tokenVal_GLOBAL;
	this.tokenValsRecord = {};
	for (let i = 0; i < this.tokenVals.length; i++) {
		this.tokenValsRecord[this.tokenVals[i]] = i;
	}
	this.tokenEmbeddings = tokenEmbeddings_GLOBAL;
	this.vocabSize = this.tokenEmbeddings.length;
	const vocabSize = this.vocabSize;

	this.setupInference = (sequenceLength) => {
		L = sequenceLength;
		this.setShaders();
		this.setTransformers();

		this.tokenEmbeddings = this.load_token_embeddings_WEBGPU(this.tokenEmbeddings);
		this.rmsGamma3 = this.load_rms_weights_WEBGPU(rmsGamma3_GLOBAL);

		globalThis.initFastTransformerBuffers(this, dimensions, heads, L, NetworkMeta.ffnDim, NetworkMeta.numTransformers, vocabSize);
	}

	this.encodeTokenBatch = (firstIndex, tokenCount, selectNextToken) => {
		if (tokenCount <= 0) {
			return;
		}

		globalThis.commandEncoder = this.webgpuDevice.createCommandEncoder();
		globalThis.passEncoder = commandEncoder.beginComputePass();

		for (let tokenOffset = 0; tokenOffset < tokenCount; tokenOffset++) {
			globalThis.setFastInferencePosition(this, firstIndex + tokenOffset);
			this.inputTokenEmbeddings = this.setInputTokenEmbeddings_WEBGPU(this.tokenEmbeddings);
			this.transformToGetNextToken(selectNextToken);
		}

		globalThis.passEncoder.end();
		this.webgpuDevice.queue.submit([globalThis.commandEncoder.finish()]);
	}

	this.beginGeneration = (queryArr) => {
		if (!Array.isArray(queryArr) || queryArr.length === 0) {
			throw new Error('Fast inference requires at least one context token.');
		}

		const contextLength = Math.min(queryArr.length, L);
		const tokenIndices = new Uint32Array(L + 1);
		for (let tokenIndex = 0; tokenIndex < contextLength; tokenIndex++) {
			const vocabIndex = this.tokenValsRecord[queryArr[tokenIndex]];
			if (vocabIndex === undefined) {
				throw new Error(`Context token at position ${tokenIndex} is not in the vocabulary.`);
			}
			tokenIndices[tokenIndex] = vocabIndex;
		}

		globalThis.initializeFastTokenIndices(this, tokenIndices);

		// Fill K/V cache for every context token except the last. The last context
		// token is the first input processed by generateTokenBatch().
		const prefillTokenCount = contextLength - 1;
		const prefillBatchSize = 16;
		for (let firstIndex = 0; firstIndex < prefillTokenCount; firstIndex += prefillBatchSize) {
			this.encodeTokenBatch(firstIndex, Math.min(prefillBatchSize, prefillTokenCount - firstIndex), false);
		}

		this.nextInferenceIndex = contextLength - 1;
		return contextLength;
	}

	this.hasMoreTokens = () => this.nextInferenceIndex < L;

	this.generateTokenBatch = async (requestedTokenCount = 5) => {
		if (!Number.isInteger(this.nextInferenceIndex)) {
			throw new Error('Call beginGeneration() before generateTokenBatch().');
		}

		const tokenCount = Math.min(requestedTokenCount, L - this.nextInferenceIndex);
		if (tokenCount <= 0) {
			return new Uint32Array();
		}

		const readbackStartIndex = this.nextInferenceIndex + 1;
		this.encodeTokenBatch(this.nextInferenceIndex, tokenCount, true);
		this.nextInferenceIndex += tokenCount;

		return globalThis.readFastTokenIndexRange(this, readbackStartIndex, tokenCount);
	}

	this.transformToGetNextToken = (selectNextToken) => {
		let nextTransformerInput = this.inputTokenEmbeddings;

		for (let tIndex = NetworkMeta.numTransformers - 1; tIndex >= 0; tIndex--) {
			nextTransformerInput = this.transformers[tIndex].applyTransformChainToInputs(nextTransformerInput);
		}

		if (selectNextToken) {
			const transforStackOutputPostRMS3 = this.transformers[0].RMSNorm(nextTransformerInput, this.rmsGamma3, lmNetwork.preBuffers.rms3Sum, lmNetwork.preBuffers.rms3);
			const logitScores = this.matMul_vocab_WEBGPU(this.tokenEmbeddings, transforStackOutputPostRMS3);
			this.selectNextToken_WEBGPU(logitScores);
		}
	}
}
