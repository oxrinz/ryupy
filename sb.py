import ryupy

cpu_tensor1 = ryupy.cuda.tensor(15)
cpu_tensor1.print_info()

cpu_tensor2 = ryupy.cuda.tensor(10)
cpu_tensor2.print_info()

cpu_tensor3 = cpu_tensor1 + cpu_tensor2 + cpu_tensor1 + cpu_tensor2 
cpu_tensor3.print_info()