import serial
import csv

# Set up the serial connection (update 'COM3' to your port)
ser = serial.Serial('COM6', 9600, timeout=1)

# Open a CSV file to write the data
with open('/Datasets/ultrasonic_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y', 'Distance'])

    while True:
        try:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:
                print(line)
                # Split the line into X, Y, and Distance
                x, y, distance = map(float, line.split(','))
                # Write the data to the CSV file
                writer.writerow([x, y, distance])
        except KeyboardInterrupt:
            break

ser.close()
