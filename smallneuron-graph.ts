import * as sn from './smallneuron';
import { DirectedGraph } from 'graphology';
import * as dag from 'graphology-dag';

// Metadata defining the model externally
export class ModelMetadata {
    inputWidth: number;
    outputWidth: number;
  
    // Hyperparameters
    learningRate: number;
    lossFunction: 'meanSquaredError';
    epochCount: number;
  
    constructor(inputWidth: number, outputWidth: number, learningRate: number, activationType: 'linear'|'relu'|'sigmoid'|'tanh', lossFunction: 'meanSquaredError', epochCount: number) {
      this.inputWidth = inputWidth;
      this.outputWidth = outputWidth;
      this.learningRate = learningRate;
      this.lossFunction = lossFunction;
      this.epochCount = epochCount;
    }
  }
  
export class GraphModel {

    metadata: ModelMetadata;
    graph: DirectedGraph;

    inputs: sn.Tensor[];
    outputs: sn.Tensor[];



    constructor(md: ModelMetadata, graph: DirectedGraph) {
        this.metadata = md;

        // validate graph
        if(dag.hasCycle(graph)) throw new Error("GraphModel Creation Error: Provided DAG is not acyclic (cycles found).");
        this.graph = graph;

        // create io tensors, zero-ed out at the start
        this.inputs = [];
        for(let i = 0; i < this.metadata.inputWidth; i++) {
            this.inputs.push(sn.tensor(0));
        }

        // map out graph nodes to tf layers
        const nodeToTensorMap = new Map<string, sn.Tensor | sn.Tensor[]>();

        // parse nodes into neurons and layers
        dag.forEachNodeInTopologicalOrder(this.graph, (node, attributes) => {

            // if its a neuron or layer node get dependencies and add to model
            if(node.includes('NEURON') || node.includes('LAYER')) {
                
                // filter edges to only the edges that end at this node
                const edgesToNode = this.graph.filterEdges(edge => 
                this.graph.target(edge) == node
                );

                // get all nodes that have an edge to the current neuron
                const srcNodes = edgesToNode.map(edge => this.graph.source(edge));
                // get all layers associated with nodes in srcNodes
                let srcNeurons: sn.Tensor[] = [];
                srcNodes.forEach(srcNode => {
                    // get node from node to layer map or from inputs
                    const srcNeuron = nodeToTensorMap.get(srcNode);

                    // nullcheck
                    if(srcNeuron == undefined) {
                        throw new Error("GraphModel Creation Error: Neuron dependency undefined, DAG neuron requested before creation.");
                    } else {
                        srcNeuron instanceof Array ? srcNeuron.forEach(n => srcNeurons.push(n)) : srcNeurons.push(srcNeuron);
                    }
                
                });

                // create new neuron from layer and sources
                if(node.includes('NEURON')) {
                    const currentNeuron: sn.Neuron = sn.neuron(srcNeurons.length, attributes.activation, attributes.nobias);

                    nodeToTensorMap.set(node, currentNeuron.call(srcNeurons));
                } else if(node.includes('LAYER')) {
                    const currentLayer: sn.Layer = sn.layer(srcNeurons.length, attributes.output_width, attributes.activation, attributes.nobias);

                    nodeToTensorMap.set(node, currentLayer.call(srcNeurons));
                }

            }

        });

        // after parsing all neurons and layers filter to nodes without edges starting from them i.e. output nodes
        const outputNodes = this.graph.nodes().filter(node =>
            !this.graph.edges().some(edge => this.graph.source(edge) == node)
        );

        // assign output tensors
        this.outputs = outputNodes.flatMap(outputNode => nodeToTensorMap.get(outputNode) ?? sn.tensor(0));

    }
}
