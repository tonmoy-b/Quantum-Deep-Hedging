from src import quantum_architecture as q_arch
#from src import architecture as classical_arch

if __name__ == "__main__":
    print("toffey!")
    device = q_arch.get_quantum_device()
    q_model = q_arch.full_train_loop()
    # classical_arch.full_train()
