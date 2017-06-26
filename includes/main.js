//main function
$(document).ready(function () {
	'use strict';

	var debug = false, //logs to console
		verbose = false, //writes messages to screen
		training = true, //are we in training mode?
		num_inputs = 4, //input node count
		num_hidden = 3, //hidden node count, 
		num_outputs = 3,  //output node count
		alpha = 1.0, //learning rate
		max_iterations = 2000,
		input_data_file = "/data/inputdata.js",
		target_data_file = "/data/targetdata.js",
		testing_data_file = "/data/testdata.js";

	/*** HELPER FUNCTIONS ***/
	//returns an array filled with zeros
	var CreateArray = function (length) {
			var arr = [], a;
			for (a = 0; a < length; a++) {
					arr[a] = 0;
			}
			return arr;
	};

	//returns an array of arrays filled with zeros
	var Create2DArray = function (rows, columns) {
		/*
			example output
			|w03 w13 w23|
			|w02 w12 w22|
			|w01 w11 w12|
		*/
		var rowarray = [], colarray = [], j, k;
		//define the columns array
		for (k = 0; k < columns; k++) {
			colarray[k] = 0;
		}
		//for each row add an array of columns
		for (j = 0; j < rows; j++) {
			rowarray[j] = colarray.slice();
		}
		//result
		return rowarray;
	};

	//returns random value between two numbers
	var GetRandomArbitrary = function (min, max) {
			if (debug) console.log('GetRandomArbitrary');

			return Math.random() * (max - min) + min;
	};

	//applies a squashing function which results in a value 0 - 1
	var Squash = function (value) {
			//applies binary sigmoid function
			//if (debug) console.log('Squash');
			return (1.0 / (1 + Math.exp(-1 * value)));
	};

	//derivative of squash
	var SquashPrime = function (value) {
			//applies derivative of squash - f'(x)
			//if (debug) console.log('SquashPrime');
			var squashedvalue = Squash(value);
			return (squashedvalue * (1.0 - squashedvalue));
	};

	//test stopping condition
	var TestStoppingCondition = function (iterator) {
			if (iterator < max_iterations) {
					return false;
			}
			else {
					return true;
			}
	};

	//return and initializes weight matrix
	var InitWeights = function (rows, cols) {
		/*
			example array with multiple rows and cols
			|w03 w13 w23|
			|w02 w12 w22|
			|w01 w11 w12|			
		*/	
		//if (debug) console.log('InitWeights');
		var row, col, weights = Create2DArray(rows, cols);

		for (row = 0; row < rows; row++) {
			for (col = 0; col < cols; col++) {
				weights[row][col] = GetRandomArbitrary(-0.5, 0.5);
			}
		}
		return weights;
	};

	var ProcessInputData = function(data){
		//return a promise
		return new Promise(function(resolve, reject) {
			//do the work
			var i = 0, node={}, arr=[], stop = false;
			try {
				data = JSON.parse(data);
				while (!stop) {
					node = data[i];
					arr[i] = Object.keys(node).map(function (key) { return node[key]; });
					//arr[i] = [node.val0, node.val1];
					//if all nodes processed
					if (++i >= data.length) stop = true;
					node = {};
				};
				if (debug) console.log('resolving ProcessInputData');
				//rsolve the promise with the data
				resolve(arr);
			} catch (error) {
				//reject the promise with the error
				reject(Error(error));
			}
		});
	};

	var ProcessTargetData = function(data) {
		//return a promise
		return new Promise(function(resolve, reject) {
			//do the work
			var i = 0, node={}, arr=[], stop = false;
			try {
				data = JSON.parse(data);
				while (!stop) {
					node = data[i];
					arr[i] = Object.keys(node).map(function (key) { return node[key]; });
					//arr[i] = [node.val0, node.val1];
					//if all nodes processed
					if (++i >= data.length) stop = true;
					node = {};
				};
				if (debug) console.log('resolving ProcessTargetData');
				//resolve and return array 
				resolve(arr);
			} catch (error) {
				//reject with the error
				reject(Error(error));
			}	
		});
	};

	var WriteToScreen = function(message){
		console.log('set timeout message');
		var i, $oa, $div;
		$oa = $("#output-area");
		for (i = 0; i < message.length; i++) {
			$div = $("<div class='message-text'>");
			$div.text(message[i]);
			$oa.append($div);
		}
	};

	//outputs to screen
	var OutputMessage = function(message) {
		console.log('outputmessage');
		setTimeout(WriteToScreen(arguments), 0);
	};

	var ProcessError = function(err) {
		console.log(err);
		OutputMessage(err);
	};
	
	/*** END HELPER FUNCTIONS ***/

	/*** NN VARIABLES ***/
	var input_vectors = [];
	var input_vector = CreateArray(num_inputs);

	var target_vectors = [];
	var target_vector = CreateArray(num_outputs);

	var test_vectors = [];
	var test_vector = CreateArray(num_inputs);


	//weight matrices with bias in position 0
	var weights_input_to_hidden = Create2DArray(num_hidden, num_inputs + 1);
	var delta_weights_input_to_hidden = Create2DArray(num_hidden, num_inputs + 1);
	var weights_hidden_to_output = Create2DArray(num_outputs, num_hidden + 1);
	var delta_weights_hidden_to_output = Create2DArray(num_outputs, num_hidden + 1);

	//used to hold the pre-squashed vals	
	var hidden_net_values = CreateArray(num_hidden);
	var output_net_values = CreateArray(num_outputs);
	//used to hold the squashed values
	var hidden_output_values = CreateArray(num_hidden);  
	var output_output_values = CreateArray(num_outputs);
	//error between outputs and target
	var hidden_error_terms = CreateArray(num_hidden);
	var output_error_terms = CreateArray(num_outputs);
	/*** END NN VARIABLES ***/

	//declare the network object
	var Network = function () { 
		if (this instanceof Network) {
				this.callBack = {};
				this.errCallBack = {};
		} else {
				return new Network();
		}
	};
	Network.prototype.GetCallBack = function () {
			return this.callBack;
	};
	Network.prototype.SetCallBack = function (val) {
			this.callBack = val;
	};
	Network.prototype.GetErrCallBack = function () {
			return this.errCallBack;
	};
	Network.prototype.SetErrCallBack = function (val) {
			this.errCallBack = val;
	};

	/*** FEED FORWARD METHODS ***/
	//multiplies inputs by weights - weight row identified by node index
	Network.prototype.CalculateNetSum = function (inputs, weights, numinlayer) {
			/*
					example hidden to output weights
					|w03 w13 w23|
					|w02 w12 w22|
					|w01 w11 w12|

					example inputs
					[i1 i2 i3 i4 i5]
			*/
			//if (debug) console.log('CalculateNetSums');
			var total = 0, h, i, bias = 0, outputs = [], wts = [];
			
			//for each row in weights
			for (i = 0; i < numinlayer; i++){
				wts = weights[i].slice(1);
				bias = weights[i][0];
				total = 0;
				//total of inputs * weights to node 
				for(h = 0; h < inputs.length; h++){
					//sum of inputs * weight
					total = total + inputs[h] * wts[h];
				}
				//add in bias and make net value for output
				outputs[i] = total + bias;
			}

			return outputs;
	};

	//applies squash function to each value in values
	Network.prototype.SquashValues = function (values, squash_function) {
			/*
					example values
					[i1 i2 i3 i4 i5]
			*/
			//if (debug) console.log('SquashValues');
			var outputs = [], index;

			for (index = 0; index < values.length; index++) {
					outputs[index] = (squash_function(values[index]));
			}
			return outputs;
	};

	/*** BACKPROP METHODS ***/
	//multiply learning rate by errorterm and squashed output of previous layer
	Network.prototype.CalculateErrorTermsOutput = function (outputs, targets, netsumsoutput, squashprime) {
			/*	
					example inputs
					[t1 t2 t3 t4 t5]

					example netsums
					[0.3 0.15 0.56]

					example outputs
					[0.1 0.39 0.87]

			*/
			//if (debug) console.log('CalculateErrorTermsOutput');
			var errors = [], index;

			for (index = 0; index < targets.length; index++) {
					errors[index] = (targets[index] - outputs[index]) * (squashprime(netsumsoutput[index]));
			}

			return errors;
	};

	//calculate the delta weights between hidden and output
	Network.prototype.CalculateDeltaWeightOutputHidden = function (outputhiddenweights, hiddenoutput, errorterm, alpha, numhidden, numoutput) {
			/*
					example hidden to output weights
					|w03 w13 w23|
					|w02 w12 w22|
					|w01 w11 w12|

					example output error terms
					[e1 e2 e3]

					example output from hidden layer
					[0.1 0.39]

			*/
			//if (debug) console.log('CalculateDeltaWeightOutputHidden');
			//return 2d array with slots for delta weights and delta bias
			var dhow = Create2DArray(numoutput, numhidden + 1);

			//weights calculation - first iterator starts at one to avoid bias
			//for every row
			for (var row = 0; row < numoutput; row++) {
					//for every column
					for (var col = 0; col < numhidden + 1; col++) {
							//first column is bias
							if (col > 0) {
									dhow[row][col] = alpha * errorterm[row] * hiddenoutput[col - 1];
							}
					}
			}

			//bias calculation
			for (row = 0; row < numoutput; row++) {
					dhow[row][0] = alpha * errorterm[row];
			}

			return dhow;
	};

	Network.prototype.CalculateErrorTermsHidden = function (errortermsoutput, weightshiddenoutput, 
																													netvalueshidden, squashprime, numhidden, 
																													numoutput) {
			/*
					example hidden to output weights
					|w03 w13 w23|
					|w02 w12 w22|
					|w01 w11 w12|

					example output error terms
					[e1 e2 e3]

					example net values from hidden layer
					[0.1 0.39]

					example hiddenerror
					[0.1 0.2]

			*/

			//if (debug) console.log('CalculateErrorTermsHidden');
			var hiddenerror = []; var sum = 0, row, col;

			//the plus 1 is to account for bias		
			for (col = 0; col < numhidden + 1; col++) {
					//ignore bias
					if (col > 0) {
							sum = 0;
							for (row = 0; row < numoutput; row++) {
									//first column is bias
									sum = sum + errortermsoutput[row] * weightshiddenoutput[row][col];
							}
							hiddenerror[col - 1] = sum * squashprime(netvalueshidden[col - 1]);
					}
			}

			return hiddenerror;
	};

	Network.prototype.CalculateDeltaWeightHiddenInput = function (weightsinputhidden, errortermshidden, inputs, alpha, numinput, numhidden) {
			//return 2d array with delta weights and delta bias
			/*
					example input to hidden weights
					|v02 v12 v22 v32|
					|v01 v11 v21 v31|

					example hidden error terms
					[e1 e2]

					example inputs
					[0.1 0.39 0.9]
			*/
			//if (debug) console.log('CalculateDeltaWeightHiddenInput');
			//return 2d array with slots for delta weights and delta bias
			var dihw = Create2DArray(numhidden, numinput + 1);
			var row, col;
			//weights calculation - first iterator starts at one to avoid bias
			//for every row
			for (row = 0; row < numhidden; row++) {
					//for every column
					for (col = 0; col < numinput + 1; col++) {
							//first column is bias
							if (col > 0) {
									dihw[row][col] = alpha * errortermshidden[row] * inputs[col - 1];
							}
					}
			}

			//bias calculation
			for (row = 0; row < numhidden; row++) {
					dihw[row][0] = alpha * errortermshidden[row];
			}

			return dihw;
	};

	Network.prototype.CalulateUpdatedWeights = function (weights, deltaweights, rows, columns) {
			/*
					example weights
					|w03 w13 w23|
					|w02 w12 w22|
					|w01 w11 w12|

					example delta weights
					|dw03 dw13 dw23|
					|dw02 dw12 dw22|
					|dw01 dw11 dw12|
			*/
			//if (debug) console.log('CalulateUpdatedWeights');
			var output = Create2DArray(rows, columns + 1);
			var row, col;
			for (row = 0; row < rows; row++) {
					for (col = 0; col < columns + 1; col++) {
							output[row][col] = weights[row][col] + deltaweights[row][col];
					}
			}

			return output;
	};

	Network.prototype.GetDataAsync = function(url){
		//if (debug) console.log('Network.GetDataAsync');
		// Return a new promise - taken from developers.google.com
		return new Promise(function(resolve, reject) {
			// Do the usual XHR stuff
			var req = new XMLHttpRequest();
			req.open('GET', url);

			//this fires when doument loads
			req.onload = function() {
				// This is called even on 404 etc
				// so check the status
				if (req.status == 200) {
					if (debug) console.log('resolving GetDataAsync');
					// Resolve the promise with the response text
					resolve(req.response);
				}
				else {
					// Otherwise reject with the status text
					// which will hopefully be a meaningful error
					reject(Error(req.statusText));
				}
			};

			// Handle network errors
			req.onerror = function() {
				reject(Error("Network Error"));
			};

			// Make the request
			req.send();
		});
	};
	
	Network.prototype.GetData = function (url, data, callBack, errCallBack) {
		/* arguments
		 * url = the url from where to get the data
		 * data = json object with parameters
		 * callBack = pointer to function to call upon success
		 * errCallBack = pointer to error handler
		 */
		//if (debug) console.log("Network.GetData");

		this.SetCallBack(callBack);
		this.SetErrCallBack(errCallBack);

		var self = this;
		data = data || {};
		$.ajax({
				method: "GET",
				url: url,
				data: data,
				dataType: "script",
				beforeSend: function () {

				},
				success: function (data) {
						var func = self.GetCallBack();
						func.call(null, data);
				},
				error: function (err) {
						var func = self.GetErrCallBack();
						func.call(null, err);
				}
		});
	};
	
	var net = new Network();

	//init weights to between -0.5 and 0.5 - add one to col param for bias
	weights_input_to_hidden = InitWeights(num_hidden, num_inputs + 1);
	weights_hidden_to_output = InitWeights(num_outputs, num_hidden + 1);
	
	//get the promises returned by the Get operation
	var input_vectors_promise = net.GetDataAsync(input_data_file);
	var target_vectors_promise = net.GetDataAsync(target_data_file);
  
	//want to wait until both get operations are fulfilled 
	Promise.all([input_vectors_promise, target_vectors_promise]).then(function(values){
		var process_input_vectors_promise = ProcessInputData(values[0]);
		var process_target_vectors_promise = ProcessTargetData(values[1]);
		//wait until we've processed the data
		Promise.all([process_input_vectors_promise, process_target_vectors_promise]).then(function(values){
			input_vectors = values[0];
			target_vectors = values[1];
			//now run the training
			RunTrainingProcess();

			//get the test data
			net.GetDataAsync(testing_data_file).then(function(value){
				//process the values
				ProcessInputData(value).then(function(value){
					test_vectors = value;
					RunMatchingProcess();
				});
			});
		});
	}, function(error){ 
		console.log(error)
	});
	
	var RunTrainingProcess = function(){
		var stop = false;
		var stop_index = 1;
		var input_vector_index;
		if (verbose) OutputMessage('starting weights ihw: ', weights_input_to_hidden); 
		if (verbose) OutputMessage('starting weights how: ', weights_hidden_to_output); 

		//loop while stopping condition false
		while (stop===false) {
			if (verbose) OutputMessage('*****************************************');
			if (verbose) OutputMessage('************ iteration: ' + stop_index + ' ************'); 
			if (verbose) OutputMessage('*****************************************');
			//loop thru input vectors
			for (input_vector_index=0; input_vector_index < input_vectors.length; input_vector_index++){
				///////get data///////
				input_vector = input_vectors[input_vector_index];
				target_vector = target_vectors[input_vector_index];

				///////dsiplay data///////
				if (verbose) OutputMessage('\n');
				if (verbose) OutputMessage('----------------- input index: ' + input_vector_index + '-----------------'); 
				if (verbose) OutputMessage('--inputs: ', input_vector); 
				if (verbose) OutputMessage('--targets: ', target_vector); 

				///////feedforward///////
				hidden_net_values = net.CalculateNetSum(input_vector, weights_input_to_hidden, num_hidden);
				//pass in the net values and the squash function
				hidden_output_values = net.SquashValues(hidden_net_values, Squash); //replace the net_values array
				output_net_values = net.CalculateNetSum(hidden_output_values, weights_hidden_to_output, num_outputs);
				output_output_values = net.SquashValues(output_net_values, Squash); //replace the hidden output array with squashed values
				if (verbose) OutputMessage('--output values: ', output_output_values);

				///////backpropogate the error///////
				//calculate error terms output layer
				output_error_terms = net.CalculateErrorTermsOutput(output_output_values, target_vector, output_net_values, SquashPrime);
				if (verbose) OutputMessage('--output error terms: ', output_error_terms);
				//calculate weight changes between hidden and output
				delta_weights_hidden_to_output = net.CalculateDeltaWeightOutputHidden(weights_hidden_to_output, hidden_output_values, output_error_terms, alpha, num_hidden, num_outputs);
				//calculate error terms hidden layer
				hidden_error_terms = net.CalculateErrorTermsHidden(output_error_terms, weights_hidden_to_output, hidden_net_values, SquashPrime, num_hidden, num_outputs);
				//calculate weight changes between input and hidden
				delta_weights_input_to_hidden = net.CalculateDeltaWeightHiddenInput(weights_input_to_hidden, hidden_error_terms, input_vector, alpha, num_inputs, num_hidden);

				//apply weights changes
				weights_hidden_to_output = net.CalulateUpdatedWeights(weights_hidden_to_output, delta_weights_hidden_to_output, num_outputs, num_hidden);
				weights_input_to_hidden = net.CalulateUpdatedWeights(weights_input_to_hidden, delta_weights_input_to_hidden, num_hidden, num_inputs);
				if (verbose) OutputMessage('updated weights ihw: ', weights_input_to_hidden); 
				if (verbose) OutputMessage('updated weights how: ', weights_hidden_to_output); 
			} //end for

			//test stopping condition
			stop_index++;
			stop = TestStoppingCondition(stop_index);
		}
	}
	
	var RunMatchingProcess = function(){
		var test_vector_index;
		OutputMessage('*****************************************');
		OutputMessage('************ test iterations ************'); 
		OutputMessage('*****************************************');

		//loop thru input vectors
		for (test_vector_index=0; test_vector_index < test_vectors.length; test_vector_index++){
			///////get data///////
			test_vector = test_vectors[test_vector_index];

			///////display data///////
			OutputMessage('----------------- test index: ' + test_vector_index + '-----------------'); 
			OutputMessage('--inputs: ', test_vector); 

			///////feedforward///////
			hidden_net_values = net.CalculateNetSum(test_vector, weights_input_to_hidden, num_hidden);
			hidden_output_values = net.SquashValues(hidden_net_values, Squash); //replace the net_values array

			output_net_values = net.CalculateNetSum(hidden_output_values, weights_hidden_to_output, num_outputs);
			output_output_values = net.SquashValues(output_net_values, Squash); //replace the hidden output array with squashed values
			OutputMessage('--output output values: ', output_output_values);
		} //end for		

	}

});