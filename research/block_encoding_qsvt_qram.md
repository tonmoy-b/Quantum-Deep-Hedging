
# Quantum Data Pipelines: Bucket-Bridge Architecture QRAM and Block-Encoding for Efficient Data Interchange between Classical and Quantum Hardware


## Motivation - The Necessity for Quantum Data Pipelines in progressing towards Quantum Advantage
In the world of R&D, Quantum Computation (QC) has been emerging from  heavy theoretical and laboratory-based proving grounds into the actual commercial space. And, in making this shift there has emerged a strong need to interface between Quantum hardware and the vast, current __classical__ tech hardware, out of which the world functions.

Any advantage of commercial adoption of QC depends, at least partially, on the establishment of efficient Quantum Data Pipelines - i.e. fast means of shipping data between currently utilized (classical) data formats and formats that are native to the QC hardware infrastructure. 

## Data Interchange Obstacles to Quantum Advantage
While, this might seem to be a trivial matter, it is not actually so. The transfer of information from a classical hardware source into a Quantum source and then back out can be computationally intensive. So much so, that being unable to constrain the execution time of such operations may render any speed gains obtained from QC processing diminished or nullified. For example, given the case where a N x N matrix is to be loaded into a Quantum computer. Should this process take $O(N^2)$ time [owing to the $N^2$ elements in the matrix], then any speed gains from efficient Quantum processing upon the matrix, as with the HHL algorithm's $O(log N)$ time complexity, is potentially nullified. Finally, the obtained results need to be sent back into the classical hardware ecosystem, further stressing the impact of any performance bottlenecks in this area. Failure to streamline this data exchange therefore can dent any advantage provided by QC.

## Solution: Quantum Random Access Memory (QRAM) - Bucket-Brigade Architecture
Element by element reading and transfer of an N x N matrix would take $O(N^2)$ time, which may render any subsequent Quantum speedup moot. QRAMs utilize the fundamental Quantum advantage of entanglement to render this process in $O(logN)$ time. QRAMs store information exactly as classical memory does. The Quantum advantage emerges during the querying of that memory, utilizing Quantum superposition to access multiple memory addresses simultaneously. The QRAM architecture being discussed specifically, is the Bucket-Brigade architecture proposed by Giovannetti, Lloyd, and Maccone. This scheme stores the information in the bottom-most leaves of a binary tree. Being a binary tree, it has a height of $n$ = $log_2(N)$. To reach any specific leaf, a signal must pass through a series of routing nodes (switches) from the root of the tree down to the leaves at the bottom of the tree. Every switch may be in one of three possible states: wait, route left or route right state. The bucket brigade architecture ensures that only $\log_2(N)$ switches along the specific path to the desired memory leaf are activated. The process of reading the data involves filling the qubits of an 'address register' for the location to be read - or a superposition of addresses to be read. These address qubits flow down the tree, conditionally turning the state of the switches from 'wait' to 'left' or 'right'. The conclusion of this action opens a superposition of paths to the specific memory registers. The address register is then entangled with the switches in the binary tree. Then, a separate register holding bus qubits is injected into the root of the tree. The bus qubits then flow down the path of the tree set by the address register previously, into the leaf nodes with the classical memory registers to be read. Having done so, the bus qubits takes on the state of the register(s), say state $|D\rangle$. This is done by an operation like the CNOT (controlled-NOT), in which the control is the held by the classical information bits with the bus qubits being targeted. This step constitutes the read operation. The bus register qubits then flow up the tree, while simultaneously resetting the switches back to the 'wait' state. This step therefore severs the entanglement between the switches and the Quantum state, thus leaving only the address and bus registers in superposition to yield a state such as: 
$$\sum_{a} \alpha_a |a\rangle_{address} \otimes |D_a\rangle_{bus}$$
A significant advantage of Bucket-Brigade methodology is that only ~log N nodes experience decoherence per query, rather than all the N nodes. Which makes the setup more physically plausible with current Quantum hardware. This increased likelihood of fidelity may even be considered more important than the speed of the operation. 

For a concrete use-case, given a financial application that needs the processing of a covariance matrix of 10,000 instruments. Naive, element-by-element, loading would need $10,000^2$ or $10^8$ for the Quantum circuit depth. If each of these are subjected to one operation, then 100 million operations will need to be done. In the practical state of current Quantum hardware, qubits would likely decohere before the completion of such a lengthy process. In the example above, the binary tree would have a height of:
$$ O(\log_2(N^2)) = O(\log_2(10^8)) \approx O(27) $$

This leads to far more plausible operations within the span of qubit coherence. However, this solution brings forth a trade-off. 

### The Design Trade-Off
Building the binary tree means the expenditure of many more switching qubits. While naive loading would result in great circuit depth, it would need less qubits. The tradeoff, hence, that between time (from the greater circuit depth of naive loading) and space (larger number of qubit needed to build the binary tree for the Bucket-Brigade Architecture). To bring the space requirement to point, it may be noted that the binary tree for 10⁸ leaves could require roughly 2 × 10⁸ routing qubits. 


## Setting: Information relevant to the Quantum Computer
QCs are highly specialized hardware that take advantage of Quantum Mechanical principles to exhibit performance improvements over conventional ('classical') hardware with regard to some computational tasks. Quantum algorithms executed on such hardware, being based on Quantum Mechanics, are inherently Linear Algebraic in nature. This renders such algorithms eminently suitable towards applications in industries like Finance, Pharmaceuticals etc. However, additional constraints come into effect due to Quantum Mechanical principles regarding the information that may be processed by QCs. A principal requirement being that information processed by the QC must be in the form of Unitary matrices. However, many real-world information, such as covariance matrices, transition matrices, payoff operators are not necessarily unitary. There arises the need therefore to transform them into a form consumable by Quantum hardware for the desired advantages of Quantum algorithms to be actualized.

## Solution: Block-Encoding Matrices 
A solution to this problem is to embed the non-unitary matrix inside a larger unitary matrix, via a method known as Block Encoding. Block Encoding uses extra 'ancilla' qubits to embed the non-Unitary matrix ($A$) inside a larger matrix ($U_A$), which actually is Unitary. A transform of the original matrix $A$ forms a block within the unitary matrix $U_A$. The transform needed consists of scaling the matrix A by a constant $\alpha$ which normalizes A by holding each element to be less than or equal to 1. The transform creates a new matrix:

$$U_A = \begin{pmatrix} A/\alpha & B \\ C & D \end{pmatrix}$$

where B, C, and D are filled with values to make the larger matrix unitary while the top-left sub-block of the matrix is the original matrix scaled to ensure that the singular values do not exceed 1, by using: 

$$\alpha \ge \|A\|_2$$

While this yields a suitable unitary matrix, it introduces the problem of isolating the original matrix eventually. This is accomplished by projecting the ancilla qubits onto the state:
 $$|0\rangle^{\otimes a}$$ 
 to retrieve the original A matrix, as per this formula:
 $$A = \alpha (\langle 0|^{\otimes a} \otimes I_n) U_A (|0\rangle^{\otimes a} \otimes I_n)$$

Conceptually, the right-hand operator:
 $$(|0\rangle^{\otimes a} \otimes I_n)$$ 
 acts as an embedding step, padding the system with zeroed ancillas to align with the top-left sub-block. After $U_A$ is applied, the left-hand operator: 
 $$(\langle 0|^{\otimes a} \otimes I_n)$$ 
 acts as an extraction step. It filters out the non-data-carrying sub-blocks ($B$, $C$, and $D$) by projecting the system back onto the  ancilla state:
 $$|0\rangle$$
Thus, finally, leaving only the desired operations of $A$.

We can see this dynamically when the unitary is applied to a Quantum state $|\psi\rangle$ padded with zeroed ancillas:
$$U_A (|0\rangle^{\otimes a} |\psi\rangle) = \frac{1}{\alpha} |0\rangle^{\otimes a} (A|\psi\rangle) + |\Phi^\perp\rangle$$
In this equation, the right-most term: 
$$|\Phi^\perp\rangle$$ 
represents the "garbage" state, the information generated when the ancillas do not measure as 0. To successfully isolate the desired state, i.e.: 
$$A|\psi\rangle$$ 
and to increase the probability of measuring the ancillas in the state: 
$$|0\rangle$$
techniques such as Oblivious Amplitude Amplification are typically employed.

## Application: Quantum Singular Value Transformation (QSVT)
A concrete benefit of ingesting data into the Quantum Computer can be seen with Quantum Singular Value Transformation (QSVT). Given a block encoding of $A$ with singular values $\sigma_i$, QSVT can implement a block encoding of $f(A)$ for any bounded polynomial function $f$, transforming each singular value $\sigma_i$ into $f(\sigma_i)$. QSVT works by interleaving applications of the block-encoded unitary $U_A$ with carefully chosen single-qubit rotations (called signal processing rotations) applied to the ancilla qubits. Each rotation is parameterized by an angle $\phi_i$, and the sequence of these angles collectively defines the polynomial transformation being applied. The key insight of QSVT is that the choice of angles $\{\phi_1, \phi_2, \dots, \phi_d\}$ (where $d$ is the polynomial degree) determines completely the specific polynomial transformation being applied. And, since the degree $d$ also determines how many times $U_A$ must be applied, circuit depth scales linearly with $d$. Finally, these transformations of QSVT leave the intermediate matrices within the block encoding framework, so that further transformations may be 'chained' along - allowing the output of one QSVT operation to serve as the block-encoded input to another, thus enabling composition of transformations.

QSVT provides a unified framework for implementing many well-studied Quantum algorithms, such as those used for matrix inversion (useful for Markowitz optimization), matrix exponentiation (transition operators or stochastic models), eigenvalue thresholding (PCA related applications), amplitude estimation (Monte Carlo acceleration for derivative pricing). 

## Conclusion: Quantum Advantage
Quantum computing has the potential for yielding exponential speedups for many problems in quantitative finance that are exceptionally demanding problems in terms of computational resources. For example: Monte Carlo pricing of exotic derivatives, portfolio optimization under constraints, credit valuation adjustment (CVA) calculations, risk aggregation across thousands of correlated instruments etc. Algorithms like HHL (for solving linear systems), Quantum amplitude estimation (for Monte Carlo acceleration), and Quantum PCA (for dimensionality reduction) have strong and well-studied theoretical foundations that suggest quadratic to exponential improvements over classical methods. It is however, important to bear in mind the needs of actual Quantum hardware to complete the picture. In particular, realizing Quantum advantage in the real world requires the complete data pipeline described above: efficient loading via QRAM, embedding via block encoding, and transformation via QSVT. 




### References
1. Gilyén, Su, Low, Wiebe — "Quantum singular value transformation and beyond" (2019) (arXiv: 1806.01838)
2. Martyn, Rossi, Tan, Chuang — "Grand Unification of Quantum Algorithms" (2021)(arXiv: 2105.02859)
3. Giovannetti, Lloyd, Maccone — "Quantum Random Access Memory" (2008)(arXiv: 0708.1879)
4. Stamatopoulos et al. — "Option Pricing using Quantum Computers" (Goldman Sachs, 2020)(arXiv: 1905.02666)
5. Dalzell et al. — "Quantum algorithms: A survey of applications and end-to-end complexities" (2023)(arXiv: 2310.03011)




