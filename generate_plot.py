f = open('./results.csv','r')
result = list()
timeaxis = list()
accaxis = list()
i = 0
for line in open('./results.csv'):
    if i == 0:
        line = f.readline()
        i += 1
        continue
    line = line.strip('\n').split(',')
    print(line[2], line[-1])
    time = float(line[-1])
    acc = float(line[2])
    timeaxis.append(time)
    accaxis.append(acc)
    i += 1
print(timeaxis)
print(accaxis)
f.close()
