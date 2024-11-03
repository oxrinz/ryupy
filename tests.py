import pytest
import ryupy
import numpy as np

def test_matrix_multiplication_2d():
    # Define two 2D tensors (matrices)
    tensor1 = ryupy.cuda.Tensor([[1, 2], [3, 4]])
    tensor2 = ryupy.cuda.Tensor([[5, 6], [7, 8]])

    # Expected result of matrix multiplication
    expected = np.array([[19, 22], [43, 50]])

    # Perform matrix multiplication
    result = tensor1 @ tensor2

    # Check if the result matches the expected output
    assert np.allclose(result.data, expected.tolist())

def test_matrix_multiplication_3d_batched():
    # Define two 3D tensors with batch dimension (2, 2, 2)
    tensor1 = ryupy.cuda.Tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    tensor2 = ryupy.cuda.Tensor([
        [[9, 10], [11, 12]],
        [[13, 14], [15, 16]]
    ])

    # Expected result of batched matrix multiplication
    expected = np.array([
        [[31, 34], [71, 78]],
        [[173, 186], [261, 282]]
    ])

    # Perform matrix multiplication
    result = tensor1 @ tensor2

    # Check if the result matches the expected output
    assert np.allclose(result.data, expected.tolist())

def test_matrix_multiplication_4d_batched():
    # Define two 4D tensors with batch dimensions (2, 2, 2, 2)
    tensor1 = ryupy.cuda.Tensor([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
    ])
    tensor2 = ryupy.cuda.Tensor([
        [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
        [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]
    ])

    # Expected result of batched matrix multiplication
    expected = np.array([
        [[[55, 58], [139, 146]], [[261, 274], [363, 380]]],
        [[[665, 686], [881, 910]], [[1157, 1194], [1483, 1530]]]
    ])

    # Perform matrix multiplication
    result = tensor1 @ tensor2

    # Check if the result matches the expected output
    assert np.allclose(result.data, expected.tolist())

def test_mismatched_dimensions():
    # Define tensors with mismatched dimensions
    tensor1 = ryupy.cuda.Tensor([[1, 2], [3, 4]])
    tensor2 = ryupy.cuda.Tensor([[1, 2]])

    # Expect an error due to dimension mismatch
    with pytest.raises(ValueError):
        _ = tensor1 @ tensor2

def test_higher_dim_multiplication_with_broadcasting():
    # Define tensors with batch broadcasting in higher dimensions
    tensor1 = ryupy.cuda.Tensor([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    tensor2 = ryupy.cuda.Tensor([[2, 0], [1, 2]])

    # Expected result using broadcasting for each batch
    expected = np.array([
        [[4, 4], [10, 8]],
        [[22, 12], [34, 16]]
    ])

    # Perform matrix multiplication with broadcasting
    result = tensor1 @ tensor2

    # Check if the result matches the expected output
    assert np.allclose(result.data, expected.tolist())

