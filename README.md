
# Hamiltonian Generation with Known Minimum Energy (HGKME)

This repository contains a Python script for generating and analyzing sparse matrices using gate information. The script performs various operations, such as applying gates to a Hamiltonian matrix, calculating statistics, and normalizing the matrix. The output includes the Hamiltonian matrix saved in `.npz` format, along with CSV files containing the matrix values and calculated statistics.

## Features

- Parse gate information from a file and apply gates to a sparse matrix.
- Generate sparse matrices based on user-specified parameters.
- Support for various gate types, including `AND`, `OR`, `XOR`, `NOR`, and more.
- Calculate and output statistical properties of diagonal and non-diagonal elements.
- Normalize the matrix using absolute maximum and Z-score normalization.
- Save the sparse matrix and its features to files for further analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HGKME.git
   cd HGKME
   ```

2. Install dependencies:
   ```bash
   pip install numpy scipy
   ```

## Usage

Run the script with the following command:

```bash
python ILmatrix_gen.py --size <matrix_size> --target_ratio <non_zero_ratio> --type <gate_type>
```

### Arguments

- `--size`: Size of the matrix (default: 10).
- `--target_ratio`: Target ratio of non-zero elements in the matrix (default: 0.01).
- `--type`: Gate type to use (default: "HAA"). Supported types include:
  - `HAA`: All gates.
  - `HZA`, `HZN`, `HNN`, `HPN`: Subsets of gates with specific configurations.
  - Single gate types like `AND`, `NAND`, `XOR`, etc.
- `--force`: Overwrite existing files (`yes` or `no`, default: `yes`).
- `--seed`: Random seed for reproducibility (default: 42).
- `--scale`: Scale factor for gate coefficients (default: 1).
- `--batch`: Batch size for intermediate updates (default: 10).

### Example

```bash
python ILmatrix_gen.py --size 20 --target_ratio 0.05 --type HNN --scale 2 --seed 123
```

## Outputs

The script generates the following outputs:

1. Sparse matrix saved in `.npz` format:
   ```
   ./matrix/<type>/<type>_N<size>_R<target_ratio>_S<scale>_matrix.npz
   ```

2. Sparse matrix values saved as a CSV file:
   ```
   ./matrix/<type>/<type>_N<size>_R<target_ratio>_S<scale>_matrix.csv
   ```

3. Statistical analysis saved as a CSV file:
   ```
   ./matrix/<type>/<type>_N<size>_R<target_ratio>_S<scale>_feature.csv
   ```

## Example Outputs

- Final sparse matrix with non-zero elements.
- Statistics such as mean, standard deviation, and range for diagonal and non-diagonal elements.
- Connection statistics (minimum, maximum, average, standard deviation).

## File Structure

- `ILmatrix_gen.py`: Main script for sparse matrix generation and analysis.
- `HamiltonianList_2-body.txt`: Example input file containing gate information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues or pull requests to improve this project.

## Author

Naoya Onizawa (naoya.onizawa.a7@tohoku.ac.jp)

## Acknowledgments

Special thanks to contributors and libraries used in this project.
