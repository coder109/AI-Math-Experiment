from Fitting import *

def main():
    # Generate some points
    number_size = 50
    parameters = [10, 50]
    x, y = generatePointsFromLine(number_size, parameters)
    noises = generateNoises(number_size)

    # Add noises
    for i in range(len(x)):
        x[i] += noises[i]
        y[i] += noises[i]

    # Using LSQ
    result_LSQ = linearFitting(x, y)
    visualizer(x, y, result_LSQ, "LSQ figure", 0)
    
    # Using RANSAC
    result_RANSAC = RANSAC(x, y)
    visualizer(x, y, result_RANSAC, "RANSAC figure", 0)

if __name__ == "__main__":
    main()
