from Fitting import *

def main():
    number_size = 50
    parameters = [3, 5, 7]
    x, y = generatePointsFromCurve(number_size, parameters)
    noises = generateNoises(number_size)

    for i in range(len(x)):
        x[i] += noises[i]
        y[i] += noises[i]

    # Using LSQ
    result_LSQ = curveFitting(x, y)
    visualizer(x, y, result_LSQ, "LSQ figure", 1)

    # Using RANSAC
    result_RANSAC = RANSAC_curve(x, y)
    visualizer_points(x, y, result_RANSAC[1], "RANSAC figure")

if __name__ == "__main__":
    main()
