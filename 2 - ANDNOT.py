# Prompt the user to enter the values for x1 and x2
x1 = [int(x) for x in input("Enter values for x1 (separated by space): ").split()]
x2 = [int(x) for x in input("Enter values for x2 (separated by space): ").split()]

# Prompt the user to enter the values for w1, w2, and theta
w1 = float(input("Enter w1: "))
w2 = float(input("Enter w2: "))
theta = float(input("Enter theta: "))

andnot = []

# Perform the ANDNOT operation for each input pair
for i in range(len(x1)): 
    s = x1[i] * w1 + x2[i] * w2

    if s >= theta:
        andnot.append(1)
    else:
        andnot.append(0)

print("Output:", andnot)

# Check if the output matches the expected output
expected_output = [0, 0, 1, 0]  # Define the expected output
if andnot == expected_output:
    print("ANDNOT is true for all inputs!")
else:
    print("ANDNOT is not true!")
