globalThis.readFloat32Buffer = async function(ctx, srcBuffer, byteSize) {
  const readBuffer = ctx.webgpuDevice.createBuffer({
    size: byteSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = ctx.webgpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(
    srcBuffer,
    0,
    readBuffer,
    0,
    byteSize
  );

  ctx.webgpuDevice.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);

  const arrayBuffer = readBuffer.getMappedRange();
  const result = new Float32Array(arrayBuffer.slice(0));

  readBuffer.unmap();
  readBuffer.destroy();

  return result;
}

globalThis.initTransformerBuffers = function(ctx, dimensions, heads, L, ffnDim, numTransformers) {
	const F = Float32Array.BYTES_PER_ELEMENT;
	const dimL = dimensions * L * F;
	const headL = heads * L * F;
	const headLL = heads * L * L * F;
	const ffnL = ffnDim * L * F;

	const makeBuf = (size) => ctx.webgpuDevice.createBuffer({
		size,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	ctx.preBuffers = {
		rms3: makeBuf(dimL),
		residual1: makeBuf(dimL),
		hadamard: makeBuf(ffnL),
		residual2: makeBuf(dimL),
	};

	ctx.preBuffersByTransformer = [];
	for (let i = 0; i < numTransformers; i++) {
		ctx.preBuffersByTransformer.push({
				rms1: makeBuf(dimL),
				rms2: makeBuf(dimL),
				matMulV: makeBuf(dimL),
				matMulK: makeBuf(dimL),
				matMulQ: makeBuf(dimL),
				ropeK: makeBuf(dimL),
				ropeQ: makeBuf(dimL),
				ktq: makeBuf(headLL),
				outputProj: makeBuf(dimL),
				colMax: makeBuf(headL),
				colSum: makeBuf(headL),
				qkRmsK: makeBuf(headL),
				qkRmsQ: makeBuf(headL),
				qkNormedK: makeBuf(dimL),
				qkNormedQ: makeBuf(dimL),
				softmax: makeBuf(headLL),
				valsAttention: makeBuf(dimL),
				gatedQueries: makeBuf(dimL),
				gatedQueriesSigmoid: makeBuf(dimL),
				queryGatedAttn: makeBuf(dimL),
				ffn1a: makeBuf(ffnL),
				ffn1b: makeBuf(ffnL),
				ffn2: makeBuf(dimL),
				silu: makeBuf(ffnL),
		});
	}
};

globalThis.executeDimDimWeightLoading = function(ctx, flatInput, dimensions, shaderCode) {
	if (!ctx.dimDim_weight_loading_pipeline) {
		const dimDimShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.dimDim_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: "auto",
			compute: { module: dimDimShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.dimDim_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.dimDim_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(dimensions / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeRmsWeightLoading = function(ctx, flatInput, dimensions, shaderCode) {
	if (!ctx.rms_weight_loading_pipeline) {
		const rmsShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.rms_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: rmsShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.rms_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.rms_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroups = Math.ceil(dimensions / 64);
	passEncoder.dispatchWorkgroups(workgroups);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeFfn1WeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.ffn1_weight_loading_pipeline) {
		const ffn1ShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.ffn1_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: ffn1ShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.ffn1_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.ffn1_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeFfn2WeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.ffn2_weight_loading_pipeline) {
		const ffn2ShaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
			
		ctx.ffn2_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: ffn2ShaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.ffn2_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.ffn2_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);

	return {
		buffer: outputBuffer,
	};
};

globalThis.executeTokenEmbeddingsWeightLoading = function(ctx, flatInput, numRows, numCols, shaderCode) {
	if (!ctx.token_embeddings_weight_loading_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.token_embeddings_weight_loading_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const inputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(inputBuffer, 0, flatInput);

	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: flatInput.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
	});

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.token_embeddings_weight_loading_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});

	const commandEncoder = ctx.webgpuDevice.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(ctx.token_embeddings_weight_loading_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(numCols / 8);
	const workgroupsY = Math.ceil(numRows / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);
	passEncoder.end();

	ctx.webgpuDevice.queue.submit([commandEncoder.finish()]);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeSetTeacherMode = function(ctx, teacherMode) {
	if (!ctx.teacherModeBuffer) {
		ctx.teacherModeBuffer = ctx.webgpuDevice.createBuffer({
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
	}
	const teacherModeVal = teacherMode ? 1 : 0;
	ctx.webgpuDevice.queue.writeBuffer(ctx.teacherModeBuffer, 0, new Uint32Array([teacherModeVal]));

	return {
		buffer: ctx.teacherModeBuffer,
	};
};

globalThis.executeSetRightEndIndex = function(ctx, rightEndIndex) {
	if (!ctx.rightEndIndexBuffer) {
		ctx.rightEndIndexBuffer = ctx.webgpuDevice.createBuffer({
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
	}
	ctx.webgpuDevice.queue.writeBuffer(ctx.rightEndIndexBuffer, 0, new Uint32Array([rightEndIndex]));

	return {
		buffer: ctx.rightEndIndexBuffer,
	};
};

globalThis.executeSetIsFirstIteration = function(ctx, isFirstIteration) {
	if (!ctx.firstIterationBuffer) {
		ctx.firstIterationBuffer = ctx.webgpuDevice.createBuffer({
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
	}
	console.log('isFirstIteration: ' + isFirstIteration);
	ctx.webgpuDevice.queue.writeBuffer(ctx.firstIterationBuffer, 0, new Uint32Array([isFirstIteration ? 1 : 0]));

	return {
		buffer: ctx.firstIterationBuffer,
	};
};

globalThis.executeSetInputTokenEmbeddings = function(ctx, indicesArray, tokenEmbeddingsBuffer, dimensions, L, shaderCode) {	
	if (!ctx.executeSetInputTokenEmbeddings_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSetInputTokenEmbeddings_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.indicesBuffer_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBuffer({
			size: L * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		ctx.outputBuffer_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBuffer({
			size: dimensions * L * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeSetInputTokenEmbeddings = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeSetInputTokenEmbeddings_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: ctx.indicesBuffer_executeSetInputTokenEmbeddings } },
				{ binding: 1, resource: { buffer: tokenEmbeddingsBuffer } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 3, resource: { buffer: ctx.outputBuffer_executeSetInputTokenEmbeddings } },
			],
		});	
	}

	ctx.webgpuDevice.queue.writeBuffer(ctx.indicesBuffer_executeSetInputTokenEmbeddings, 0, indicesArray);

	passEncoder.setPipeline(ctx.executeSetInputTokenEmbeddings_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeSetInputTokenEmbeddings);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.outputBuffer_executeSetInputTokenEmbeddings };
};

globalThis.executeRMSNorm = function(ctx, xInputsBuffer, rmsGammaBuffer, dimensions, L, shaderCode, rmsOutputBuffer) {
	if (!ctx.executeRMSNorm_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeRMSNorm_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeRMSNorm_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: xInputsBuffer } },
			{ binding: 1, resource: { buffer: rmsGammaBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 4, resource: { buffer: rmsOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeRMSNorm_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: rmsOutputBuffer };
};

globalThis.executeColSumRMSNormByHead = function(ctx, xInputsBuffer, heads, L, shaderCode, outputBuffer) {
	if (!ctx.executeColSumRMSNormByHead_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.executeColSumRMSNormByHead_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeColSumRMSNormByHead_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: xInputsBuffer } },
			{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 2, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 3, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeColSumRMSNormByHead_pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	// Shader is @workgroup_size(1, 8): x covers one column per workgroup, y covers 8 heads per workgroup.
	const workgroupsX = globalThis.postFirstIteration ? 1 : globalThis.LSequence;
	const workgroupsY = Math.ceil(heads / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: outputBuffer };
};

globalThis.executeRMSNormByHead = function(ctx, xInputsBuffer, denominatorBuffer, rmsGammaBuffer, heads, headDim, L, shaderCode, outputBuffer) {
	if (!ctx.executeRMSNormByHead_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.executeRMSNormByHead_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeRMSNormByHead_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: xInputsBuffer } },
			{ binding: 1, resource: { buffer: denominatorBuffer } },
			{ binding: 2, resource: { buffer: rmsGammaBuffer } },
			{ binding: 3, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 4, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 5, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeRMSNormByHead_pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	// Shader is @workgroup_size(1, 8, 8): x -> one column per workgroup, y -> 8 heads, z -> 8 head-dim rows.
	const workgroupsX = globalThis.postFirstIteration ? 1 : globalThis.LSequence;
	const workgroupsY = Math.ceil(heads / 8);
	const workgroupsZ = Math.ceil(headDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return { buffer: outputBuffer };
};

globalThis.executeMatMul_dim_L_dim_dim = function(ctx, aBuffer, bBuffer, dimensions, L, shaderCode, outputBuffer) {
	if (!ctx.executeMatMul_Dim_L_Dim_Dim_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMul_Dim_L_Dim_Dim_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMul_Dim_L_Dim_Dim_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: aBuffer } },
			{ binding: 1, resource: { buffer: bBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 4, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMul_Dim_L_Dim_Dim_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: outputBuffer };
};

globalThis.executeLoadPrecomputedTheta = function(ctx, flatData, rows, cols) {
	const bufferSize = flatData.byteLength;
	const outputBuffer = ctx.webgpuDevice.createBuffer({
		size: bufferSize,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	
	ctx.webgpuDevice.queue.writeBuffer(outputBuffer, 0, flatData);
	
	return {
		buffer: outputBuffer,
	};
};

globalThis.executeRoPE = function(ctx, inputBuffer, thetaBuffer, headDim, heads, L, shaderCode, ropeOutputBuffer) {
	if (!ctx.executeRoPE_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeRoPE_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeRoPE_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: thetaBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 4, resource: { buffer: ropeOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeRoPE_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(headDim / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return {
		buffer: ropeOutputBuffer,
	};
};

globalThis.executeDebugIsFirstIteration = function(ctx, shaderCode) {
	if (!ctx.executeDebugIsFirstIteration_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.executeDebugIsFirstIteration_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.debugIsFirstIterationOutputBuffer = ctx.webgpuDevice.createBuffer({
			size: 256 * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		});

		ctx.debugIsFirstIterationZeroes = new Float32Array(256);
	}

	ctx.webgpuDevice.queue.writeBuffer(ctx.debugIsFirstIterationOutputBuffer, 0, ctx.debugIsFirstIterationZeroes);

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeDebugIsFirstIteration_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 1, resource: { buffer: ctx.debugIsFirstIterationOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeDebugIsFirstIteration_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(256 / 8));

	return { buffer: ctx.debugIsFirstIterationOutputBuffer };
};

globalThis.executeKtQ = function(ctx, heads, shaderCode, tIndex) {
	if (!ctx.executeKtQ_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeKtQ_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeKtQ_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ropeK } },
			{ binding: 1, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ropeQ } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },			
			{ binding: 4, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ktq } },
		],
	});

	passEncoder.setPipeline(ctx.executeKtQ_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(globalThis.LSequence / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return { buffer: ctx.preBuffersByTransformer[tIndex].ktq };
};

globalThis.executeColMax = (ctx, heads, L, shaderCode, tIndex) => {
	if (!ctx.executeColMax_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeColMax_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeColMax_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ktq } },
			{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 2, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 3, resource: { buffer: ctx.preBuffersByTransformer[tIndex].colMax } },
		],
	});

	passEncoder.setPipeline(ctx.executeColMax_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(L * heads / 32));

	return { buffer: ctx.preBuffersByTransformer[tIndex].colMax };
}

globalThis.executeColSum = (ctx, heads, L, shaderCode, tIndex) => {
	if (!ctx.executeColSum_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeColSum_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeColSum_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ktq } },
			{ binding: 1, resource: { buffer: ctx.preBuffersByTransformer[tIndex].colMax } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 4, resource: { buffer: ctx.preBuffersByTransformer[tIndex].colSum } },
		],
	});

	passEncoder.setPipeline(ctx.executeColSum_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(L * heads / 32));

	return { buffer: ctx.preBuffersByTransformer[tIndex].colSum };
}

globalThis.executeSoftmaxByHead = (ctx, heads, shaderCode, tIndex) => {
	if (!ctx.executeSoftmaxByHead_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSoftmaxByHead_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeSoftmaxByHead_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ktq } },
			{ binding: 1, resource: { buffer: ctx.preBuffersByTransformer[tIndex].colMax } },
			{ binding: 2, resource: { buffer: ctx.preBuffersByTransformer[tIndex].colSum } },
			{ binding: 3, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 4, resource: { buffer: ctx.firstIterationBuffer } },						
			{ binding: 5, resource: { buffer: ctx.preBuffersByTransformer[tIndex].softmax } },
		],
	});

	passEncoder.setPipeline(ctx.executeSoftmaxByHead_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(
		Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8), 
		Math.ceil(globalThis.LSequence / 8), 
		heads
	);

	return { buffer: ctx.preBuffersByTransformer[tIndex].softmax };
};

globalThis.executeMatMulValsAttention = (ctx, headDim, heads, shaderCode, tIndex) => {
	if (!ctx.executeMatMulValsAttention_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMulValsAttention_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	ctx.executeMatMulValsAttention_bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMulValsAttention_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].matMulV } },
			{ binding: 1, resource: { buffer: ctx.preBuffersByTransformer[tIndex].softmax } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },			
			{ binding: 4, resource: { buffer: ctx.preBuffersByTransformer[tIndex].valsAttention } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMulValsAttention_pipeline);
	passEncoder.setBindGroup(0, ctx.executeMatMulValsAttention_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(headDim / 8);
	const workgroupsZ = heads;
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

	return { buffer: ctx.preBuffersByTransformer[tIndex].valsAttention };
};

// TODO: clean signature, some params not longer needed
globalThis.executeElementWiseAdd = (ctx, aBuffer, bBuffer, dimensions, L, shaderCode, outputBuffer) => {
	if (!ctx.executeElementWiseAdd_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeElementWiseAdd_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeElementWiseAdd_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: aBuffer } },
			{ binding: 1, resource: { buffer: bBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeElementWiseAdd_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);

	return { buffer: outputBuffer };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulFFN1 = (ctx, weightsBuffer, inputBuffer, ffnDim, dimensions, L, shaderCode, ffnOutputBuffer) => {
	if (!ctx.executeMatMulFFN1_pipeline) {	
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeMatMulFFN1_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMulFFN1_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },			
			{ binding: 4, resource: { buffer: ffnOutputBuffer } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMulFFN1_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);

	return { buffer: ffnOutputBuffer };
};

// TODO: clean signature, some params not longer needed
globalThis.executeSilu = (ctx, inputBuffer, ffnDim, L, shaderCode, tIndex) => {
	if (!ctx.executeSilu_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeSilu_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeSilu_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ffn1a } },
			{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 2, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 3, resource: { buffer: ctx.preBuffersByTransformer[tIndex].silu } },
		],
	});

	passEncoder.setPipeline(ctx.executeSilu_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffersByTransformer[tIndex].silu };
};

// TODO: clean signature, some params not longer needed
globalThis.executeHadamard = (ctx, aBuffer, bBuffer, ffnDim, L, shaderCode, tIndex) => {
	if (!ctx.executeHadamard_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeHadamard_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeHadamard_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.preBuffersByTransformer[tIndex].silu } },
			{ binding: 1, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ffn1b } },
			{ binding: 2, resource: { buffer: ctx.preBuffers.hadamard } },
		],
	});

	passEncoder.setPipeline(ctx.executeHadamard_pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(ffnDim / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffers.hadamard };
};

globalThis.executeSigmoid = (ctx, inputBuffer, dimensions, L, shaderCode, tIndex) => {
	if (!ctx.executeSigmoid_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.executeSigmoid_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeSigmoid_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 2, resource: { buffer: ctx.firstIterationBuffer } },
			{ binding: 3, resource: { buffer: ctx.preBuffersByTransformer[tIndex].gatedQueriesSigmoid } },
		],
	});

	passEncoder.setPipeline(ctx.executeSigmoid_pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffersByTransformer[tIndex].gatedQueriesSigmoid };
};

globalThis.executeQueryGatingHadamard = (ctx, aBuffer, bBuffer, dimensions, L, shaderCode, tIndex) => {
	if (!ctx.executeQueryGatingHadamard_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });

		ctx.executeQueryGatingHadamard_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeQueryGatingHadamard_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: aBuffer } },
			{ binding: 1, resource: { buffer: bBuffer } },
			{ binding: 2, resource: { buffer: ctx.preBuffersByTransformer[tIndex].queryGatedAttn } },
		],
	});

	passEncoder.setPipeline(ctx.executeQueryGatingHadamard_pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffersByTransformer[tIndex].queryGatedAttn };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulFFN2 = (ctx, weightsBuffer, inputBuffer, dimensions, ffnDim, L, shaderCode, tIndex) => {
	if (!ctx.executeMatMulFFN2_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeMatMulFFN2_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}

	ctx.executeMatMulFFN2_bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: ctx.executeMatMulFFN2_pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: ctx.preBuffers.hadamard } },
			{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },
			{ binding: 3, resource: { buffer: ctx.firstIterationBuffer } },			
			{ binding: 4, resource: { buffer: ctx.preBuffersByTransformer[tIndex].ffn2 } },
		],
	});

	passEncoder.setPipeline(ctx.executeMatMulFFN2_pipeline);
	passEncoder.setBindGroup(0, ctx.executeMatMulFFN2_bindGroup);
	
	const workgroupsX = Math.ceil(globalThis.postFirstIteration ? 1 : globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(dimensions / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.preBuffersByTransformer[tIndex].ffn2 };
};

// TODO: clean signature, some params not longer needed
globalThis.executeMatMulVocab = (ctx, embeddingsBuffer, inputBuffer, vocabSize, dimensions, L, shaderCode) => {
	if (!ctx.executeMatMulVocab_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeMatMulVocab_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeMatMulVocab = ctx.webgpuDevice.createBuffer({
			size: vocabSize * L * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeMatMulVocab = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeMatMulVocab_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: embeddingsBuffer } },
				{ binding: 1, resource: { buffer: ctx.preBuffers.rms3 } },
				{ binding: 2, resource: { buffer: ctx.teacherModeBuffer } },
				{ binding: 3, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 4, resource: { buffer: ctx.outputBuffer_executeMatMulVocab } },
			],
		});
	}

	passEncoder.setPipeline(ctx.executeMatMulVocab_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeMatMulVocab);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(vocabSize / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return { buffer: ctx.outputBuffer_executeMatMulVocab };
};

globalThis.executeLogitSoftmax = (ctx, logitsBuffer, vocabSize, L, shaderCode) => {
	if (!ctx.executeLogitSoftmax_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
	
		ctx.executeLogitSoftmax_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeLogitSoftmax = ctx.webgpuDevice.createBuffer({
			size: vocabSize * L * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});
		
		ctx.bindGroup_executeLogitSoftmax = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeLogitSoftmax_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: logitsBuffer } },
				{ binding: 1, resource: { buffer: ctx.teacherModeBuffer } },
				{ binding: 2, resource: { buffer: ctx.rightEndIndexBuffer } },			
				{ binding: 3, resource: { buffer: ctx.outputBuffer_executeLogitSoftmax } },
			],
		});		
	}

	passEncoder.setPipeline(ctx.executeLogitSoftmax_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeLogitSoftmax);
	
	const workgroupsX = Math.ceil(globalThis.LSequence / 8);
	const workgroupsY = Math.ceil(vocabSize / 8);
	passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY);

	return {
		buffer: ctx.outputBuffer_executeLogitSoftmax,
	};
};

globalThis.executeExtractPredictions = (ctx, softmaxBuffer, vocabSize, L, rightEndIndex, shaderCode) => {
	if (!ctx.executeExtractPredictions_pipeline) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		
		ctx.executeExtractPredictions_pipeline = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});

		ctx.outputBuffer_executeExtractPredictions = ctx.webgpuDevice.createBuffer({
			size:  vocabSize * 2 * Float32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		ctx.bindGroup_executeExtractPredictions = ctx.webgpuDevice.createBindGroup({
			layout: ctx.executeExtractPredictions_pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: softmaxBuffer } },
				{ binding: 1, resource: { buffer: ctx.rightEndIndexBuffer } },
				{ binding: 2, resource: { buffer: ctx.outputBuffer_executeExtractPredictions } },
			],
		});		
	}		

	passEncoder.setPipeline(ctx.executeExtractPredictions_pipeline);
	passEncoder.setBindGroup(0, ctx.bindGroup_executeExtractPredictions);
	
	const workgroupsX = Math.ceil(vocabSize / 64);
	passEncoder.dispatchWorkgroups(workgroupsX);

	return {
		buffer: ctx.outputBuffer_executeExtractPredictions,
	};
};

// Fast one-token inference helpers. These are intentionally separate from the
// full-sequence helpers above so index.html keeps its existing execution path.
const fastAlignTo = (value, alignment) => Math.ceil(value / alignment) * alignment;

const getFastPipeline = (ctx, key, shaderCode) => {
	if (!ctx[key]) {
		const shaderModule = ctx.webgpuDevice.createShaderModule({ code: shaderCode });
		ctx[key] = ctx.webgpuDevice.createComputePipeline({
			layout: 'auto',
			compute: { module: shaderModule, entryPoint: 'main' },
		});
	}
	return ctx[key];
};

const fastPositionResource = (ctx) => ({
	buffer: ctx.fastPositionBuffer,
	offset: ctx.fastCurrentPosition * ctx.fastPositionStride,
	size: Uint32Array.BYTES_PER_ELEMENT,
});

const fastCachePositionResource = (ctx) => ({
	buffer: ctx.fastCachePositionBuffer,
	offset: ctx.fastCurrentPosition * ctx.fastPositionStride,
	size: Uint32Array.BYTES_PER_ELEMENT,
});

const fastThetaResource = (ctx, theta) => ({
	buffer: theta.buffer,
	offset: ctx.fastCurrentPosition * theta.stride,
	size: theta.headDim * Float32Array.BYTES_PER_ELEMENT,
});

globalThis.initFastTransformerBuffers = function(ctx, dimensions, heads, L, ffnDim, numTransformers, vocabSize) {
	const F = Float32Array.BYTES_PER_ELEMENT;
	const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
	const makeBuffer = (size, usage = storageUsage) => ctx.webgpuDevice.createBuffer({ size, usage });

	ctx.preBuffers = {
		inputToken: makeBuffer(dimensions * F),
		rms3Sum: makeBuffer(F),
		rms3: makeBuffer(dimensions * F),
		residual1: makeBuffer(dimensions * F),
		hadamard: makeBuffer(ffnDim * F),
		residual2: makeBuffer(dimensions * F),
		logits: makeBuffer(vocabSize * F),
	};

	ctx.preBuffersByTransformer = [];
	for (let transformerIndex = 0; transformerIndex < numTransformers; transformerIndex++) {
		ctx.preBuffersByTransformer.push({
			rms1Sum: makeBuffer(F),
			rms1: makeBuffer(dimensions * F),
			rms2Sum: makeBuffer(F),
			rms2: makeBuffer(dimensions * F),
			matMulV: makeBuffer(dimensions * F),
			matMulK: makeBuffer(dimensions * F),
			matMulQ: makeBuffer(dimensions * F),
			ropeK: makeBuffer(dimensions * F),
			ropeQ: makeBuffer(dimensions * F),
			keyCache: makeBuffer(dimensions * L * F),
			valueCache: makeBuffer(dimensions * L * F),
			ktq: makeBuffer(heads * L * F),
			outputProj: makeBuffer(dimensions * F),
			colMax: makeBuffer(heads * F),
			colSum: makeBuffer(heads * F),
			qkRmsK: makeBuffer(heads * F),
			qkRmsQ: makeBuffer(heads * F),
			qkNormedK: makeBuffer(dimensions * F),
			qkNormedQ: makeBuffer(dimensions * F),
			softmax: makeBuffer(heads * L * F),
			valsAttention: makeBuffer(dimensions * F),
			gatedQueries: makeBuffer(dimensions * F),
			gatedQueriesSigmoid: makeBuffer(dimensions * F),
			queryGatedAttn: makeBuffer(dimensions * F),
			ffn1a: makeBuffer(ffnDim * F),
			ffn1b: makeBuffer(ffnDim * F),
			ffn2: makeBuffer(dimensions * F),
			silu: makeBuffer(ffnDim * F),
		});
	}

	ctx.fastIndicesBuffer = makeBuffer(
		(L + 1) * Uint32Array.BYTES_PER_ELEMENT,
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
	);
	ctx.fastTokenReadbackCapacity = 5;
	ctx.fastTokenReadbackBuffer = makeBuffer(
		ctx.fastTokenReadbackCapacity * Uint32Array.BYTES_PER_ELEMENT,
		GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
	);

	const positionAlignment = ctx.webgpuDevice.limits.minStorageBufferOffsetAlignment || 256;
	ctx.fastPositionStride = fastAlignTo(Uint32Array.BYTES_PER_ELEMENT, positionAlignment);
	const positionStrideWords = ctx.fastPositionStride / Uint32Array.BYTES_PER_ELEMENT;
	const positionValues = new Uint32Array((L + 1) * positionStrideWords);
	const cachePositionValues = new Uint32Array((L + 1) * positionStrideWords);
	for (let position = 0; position <= L; position++) {
		positionValues[position * positionStrideWords] = position;
		cachePositionValues[position * positionStrideWords] = position === 0 ? 0xffffffff : position - 1;
	}

	ctx.fastPositionBuffer = makeBuffer(
		positionValues.byteLength,
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	);
	ctx.fastCachePositionBuffer = makeBuffer(
		cachePositionValues.byteLength,
		GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
	);
	ctx.webgpuDevice.queue.writeBuffer(ctx.fastPositionBuffer, 0, positionValues);
	ctx.webgpuDevice.queue.writeBuffer(ctx.fastCachePositionBuffer, 0, cachePositionValues);
	ctx.fastSequenceLength = L;
};

globalThis.prepareFastPrecomputedTheta = function(ctx, flatTheta, headDim, L) {
	const alignment = ctx.webgpuDevice.limits.minStorageBufferOffsetAlignment || 256;
	const stride = fastAlignTo(headDim * Float32Array.BYTES_PER_ELEMENT, alignment);
	const strideFloats = stride / Float32Array.BYTES_PER_ELEMENT;
	const paddedTheta = new Float32Array(strideFloats * L);
	for (let position = 0; position < L; position++) {
		paddedTheta.set(flatTheta.subarray(position * headDim, (position + 1) * headDim), position * strideFloats);
	}

	const buffer = ctx.webgpuDevice.createBuffer({
		size: paddedTheta.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	ctx.webgpuDevice.queue.writeBuffer(buffer, 0, paddedTheta);
	return { buffer, stride, headDim };
};

globalThis.initializeFastTokenIndices = function(ctx, tokenIndices) {
	if (tokenIndices.byteLength !== ctx.fastIndicesBuffer.size) {
		throw new Error('Fast token index data must contain exactly L + 1 entries.');
	}
	ctx.webgpuDevice.queue.writeBuffer(ctx.fastIndicesBuffer, 0, tokenIndices);
};

globalThis.setFastInferencePosition = function(ctx, position) {
	if (!Number.isInteger(position) || position < 0 || position >= ctx.fastSequenceLength) {
		throw new RangeError(`Fast inference position ${position} is outside the K/V cache.`);
	}
	ctx.fastCurrentPosition = position;
};

globalThis.readFastTokenIndexRange = async function(ctx, startIndex, count) {
	if (startIndex < 0 || count < 0 || startIndex + count > ctx.fastSequenceLength + 1) {
		throw new RangeError('Fast token readback range is outside the L + 1 index buffer.');
	}
	if (count === 0) {
		return new Uint32Array();
	}
	if (count > ctx.fastTokenReadbackCapacity) {
		throw new RangeError(`Fast token readback supports at most ${ctx.fastTokenReadbackCapacity} indices.`);
	}

	const byteSize = count * Uint32Array.BYTES_PER_ELEMENT;
	const encoder = ctx.webgpuDevice.createCommandEncoder();
	encoder.copyBufferToBuffer(
		ctx.fastIndicesBuffer,
		startIndex * Uint32Array.BYTES_PER_ELEMENT,
		ctx.fastTokenReadbackBuffer,
		0,
		byteSize
	);
	ctx.webgpuDevice.queue.submit([encoder.finish()]);
	await ctx.fastTokenReadbackBuffer.mapAsync(GPUMapMode.READ);
	const result = new Uint32Array(ctx.fastTokenReadbackBuffer.getMappedRange(0, byteSize).slice(0));
	ctx.fastTokenReadbackBuffer.unmap();
	return result;
};

globalThis.executeFastSetInputTokenEmbeddings = function(ctx, tokenEmbeddingsBuffer, dimensions, shaderCode) {
	const pipeline = getFastPipeline(ctx, 'fastSetInputTokenEmbeddingsPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: ctx.fastIndicesBuffer } },
			{ binding: 1, resource: { buffer: tokenEmbeddingsBuffer } },
			{ binding: 2, resource: fastPositionResource(ctx) },
			{ binding: 3, resource: { buffer: ctx.preBuffers.inputToken } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: ctx.preBuffers.inputToken };
};

globalThis.executeFastRMSNorm = function(ctx, inputBuffer, gammaBuffer, dimensions, sumShaderCode, coordShaderCode, sumBuffer, outputBuffer) {
	const sumPipeline = getFastPipeline(ctx, 'fastRmsSumPipeline', sumShaderCode);
	const sumBindGroup = ctx.webgpuDevice.createBindGroup({
		layout: sumPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: sumBuffer } },
		],
	});
	passEncoder.setPipeline(sumPipeline);
	passEncoder.setBindGroup(0, sumBindGroup);
	passEncoder.dispatchWorkgroups(1);

	const coordPipeline = getFastPipeline(ctx, 'fastRmsCoordPipeline', coordShaderCode);
	const coordBindGroup = ctx.webgpuDevice.createBindGroup({
		layout: coordPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: sumBuffer } },
			{ binding: 2, resource: { buffer: gammaBuffer } },
			{ binding: 3, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(coordPipeline);
	passEncoder.setBindGroup(0, coordBindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastColSumRMSNormByHead = function(ctx, inputBuffer, heads, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastQkRmsSumPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(heads);
	return { buffer: outputBuffer };
};

globalThis.executeFastRMSNormByHead = function(ctx, inputBuffer, denominatorBuffer, gammaBuffer, dimensions, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastQkRmsCoordPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: denominatorBuffer } },
			{ binding: 2, resource: { buffer: gammaBuffer } },
			{ binding: 3, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastMatMulDimDim = function(ctx, weightsBuffer, inputBuffer, dimensions, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastMatMulDimDimPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastRoPE = function(ctx, inputBuffer, theta, headDim, heads, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastRopePipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: fastThetaResource(ctx, theta) },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(heads / 2), Math.ceil((headDim / 2) / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastSetNewValuesKeys = function(ctx, valuesBuffer, keysBuffer, dimensions, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastSetNewValuesKeysPipeline', shaderCode);
	const cache = ctx.preBuffersByTransformer[tIndex];
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: valuesBuffer } },
			{ binding: 1, resource: { buffer: keysBuffer } },
			{ binding: 2, resource: fastCachePositionResource(ctx) },
			{ binding: 3, resource: { buffer: cache.valueCache } },
			{ binding: 4, resource: { buffer: cache.keyCache } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { values: cache.valueCache, keys: cache.keyCache };
};

globalThis.executeFastKtQ = function(ctx, heads, L, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastKtQPipeline', shaderCode);
	const cache = ctx.preBuffersByTransformer[tIndex];
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: cache.keyCache } },
			{ binding: 1, resource: { buffer: cache.ropeQ } },
			{ binding: 2, resource: fastPositionResource(ctx) },
			{ binding: 3, resource: { buffer: cache.ktq } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(heads / 2), Math.ceil(L / 32));
	return { buffer: cache.ktq };
};

globalThis.executeFastSoftmaxByHead = function(ctx, heads, L, reductionShaderCode, coordShaderCode, tIndex) {
	const cache = ctx.preBuffersByTransformer[tIndex];
	const reductionPipeline = getFastPipeline(ctx, 'fastAttentionSoftmaxReductionPipeline', reductionShaderCode);
	const reductionBindGroup = ctx.webgpuDevice.createBindGroup({
		layout: reductionPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: cache.ktq } },
			{ binding: 1, resource: fastPositionResource(ctx) },
			{ binding: 2, resource: { buffer: cache.colMax } },
			{ binding: 3, resource: { buffer: cache.colSum } },
		],
	});
	passEncoder.setPipeline(reductionPipeline);
	passEncoder.setBindGroup(0, reductionBindGroup);
	passEncoder.dispatchWorkgroups(heads);

	const coordPipeline = getFastPipeline(ctx, 'fastAttentionSoftmaxCoordPipeline', coordShaderCode);
	const coordBindGroup = ctx.webgpuDevice.createBindGroup({
		layout: coordPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: cache.ktq } },
			{ binding: 1, resource: { buffer: cache.colMax } },
			{ binding: 2, resource: { buffer: cache.colSum } },
			{ binding: 3, resource: fastPositionResource(ctx) },
			{ binding: 4, resource: { buffer: cache.softmax } },
		],
	});
	passEncoder.setPipeline(coordPipeline);
	passEncoder.setBindGroup(0, coordBindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil((heads * L) / 32));
	return { buffer: cache.softmax };
};

globalThis.executeFastMatMulValsAttention = function(ctx, headDim, heads, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastMatMulValsAttentionPipeline', shaderCode);
	const cache = ctx.preBuffersByTransformer[tIndex];
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: cache.valueCache } },
			{ binding: 1, resource: { buffer: cache.softmax } },
			{ binding: 2, resource: fastPositionResource(ctx) },
			{ binding: 3, resource: { buffer: cache.valsAttention } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(heads / 2), Math.ceil(headDim / 32));
	return { buffer: cache.valsAttention };
};

globalThis.executeFastElementWiseAdd = function(ctx, leftBuffer, rightBuffer, dimensions, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastElementWiseAddPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: leftBuffer } },
			{ binding: 1, resource: { buffer: rightBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastMatMulFFN1 = function(ctx, weightsBuffer, inputBuffer, ffnDim, shaderCode, outputBuffer) {
	const pipeline = getFastPipeline(ctx, 'fastMatMulFfn1Pipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(ffnDim / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastSilu = function(ctx, inputBuffer, ffnDim, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastSiluPipeline', shaderCode);
	const outputBuffer = ctx.preBuffersByTransformer[tIndex].silu;
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(ffnDim / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastHadamard = function(ctx, leftBuffer, rightBuffer, ffnDim, shaderCode) {
	const pipeline = getFastPipeline(ctx, 'fastHadamardPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: leftBuffer } },
			{ binding: 1, resource: { buffer: rightBuffer } },
			{ binding: 2, resource: { buffer: ctx.preBuffers.hadamard } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(ffnDim / 32));
	return { buffer: ctx.preBuffers.hadamard };
};

globalThis.executeFastSigmoid = function(ctx, inputBuffer, dimensions, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastSigmoidPipeline', shaderCode);
	const outputBuffer = ctx.preBuffersByTransformer[tIndex].gatedQueriesSigmoid;
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: inputBuffer } },
			{ binding: 1, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastQueryGatingHadamard = function(ctx, leftBuffer, rightBuffer, dimensions, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastQueryGatingHadamardPipeline', shaderCode);
	const outputBuffer = ctx.preBuffersByTransformer[tIndex].queryGatedAttn;
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: leftBuffer } },
			{ binding: 1, resource: { buffer: rightBuffer } },
			{ binding: 2, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastMatMulFFN2 = function(ctx, weightsBuffer, inputBuffer, dimensions, shaderCode, tIndex) {
	const pipeline = getFastPipeline(ctx, 'fastMatMulFfn2Pipeline', shaderCode);
	const outputBuffer = ctx.preBuffersByTransformer[tIndex].ffn2;
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: weightsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 4, resource: { buffer: outputBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(dimensions / 32));
	return { buffer: outputBuffer };
};

globalThis.executeFastMatMulVocab = function(ctx, embeddingsBuffer, inputBuffer, vocabSize, shaderCode) {
	const pipeline = getFastPipeline(ctx, 'fastMatMulVocabPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: embeddingsBuffer } },
			{ binding: 1, resource: { buffer: inputBuffer } },
			{ binding: 2, resource: { buffer: ctx.preBuffers.logits } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(Math.ceil(vocabSize / 32));
	return { buffer: ctx.preBuffers.logits };
};

globalThis.executeFastVocabMax = function(ctx, logitsBuffer, shaderCode) {
	const pipeline = getFastPipeline(ctx, 'fastVocabMaxPipeline', shaderCode);
	const bindGroup = ctx.webgpuDevice.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: logitsBuffer } },
			{ binding: 1, resource: fastPositionResource(ctx) },
			{ binding: 2, resource: { buffer: ctx.fastIndicesBuffer } },
		],
	});
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(1);
	return { buffer: ctx.fastIndicesBuffer };
};
