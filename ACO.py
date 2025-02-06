import numpy as np
import random
import matplotlib.pyplot as plt


class Task:
    def __init__(self, taskId):
        self.taskId = taskId  #attribute for Task id
        self.processingTime = 0  #attribute for task's processing time

class Machine:
    def __init__(self, machineId):
        self.machineId = machineId  #attribute for machine id
        self.assignedTasks = []  #list to hold tasks assigned to this machine
        self.totalProcessingTime = 0  #total processing time for this machine

    #assign a task to the machine and update the total processing time
    def assignTask(self, task, processingTime):
        self.assignedTasks.append(task)
        task.processingTime = processingTime
        self.totalProcessingTime += processingTime

class TaskSchedulingACO:
    def __init__(self, ants, tasks, machines1, taskTimes, pheromoneEvaporation=0.1, alpha=1, beta=2):
        self.ants = ants
        self.tasks = tasks
        self.machines1 = machines1
        self.taskTimes = taskTimes
        self.pheromoneEvaporation = pheromoneEvaporation
        self.alpha = alpha
        self.beta = beta
        self.pheromoneMatrix = np.ones((tasks, machines1))  #pheromone matrix

    #ACO algorithm to find the best schedule
    def findOptimalSchedule(self, maxIterations):
        bestSchedule, bestMakespan = None, float('inf')
        worstSchedule = None
        worstMakespan = -float('inf')
        totalMakespan = 0  #calculating the average makespan
        runs = maxIterations

        #charts
        makespan_progress = []  # mintor the progress of Makespan

        bestLoad = None
        worstLoad = None
        avgLoad = []

        for _ in range(maxIterations):
            schedules, makespans, loads = [], [], []

            for _ in range(self.ants):
                schedule = self.createSchedule()
                makespan = self.calculateMakespan(schedule)
                #track load for each machine
                loads.append([machine.totalProcessingTime for machine in schedule])
                schedules.append(schedule)
                makespans.append(makespan)

                #update best makespan and best load
                if makespan < bestMakespan:
                    bestMakespan, bestSchedule = makespan, schedule
                    # Calculate load for each machine in the best schedule
                    bestLoad = [machine.totalProcessingTime for machine in bestSchedule]

                #update worst makespan and worst load
                if makespan > worstMakespan:
                    worstMakespan = makespan
                    worstSchedule = schedule
                    #calculate load for each machine in the worst schedule
                    worstLoad = [machine.totalProcessingTime for machine in worstSchedule]

                totalMakespan += makespan

            self.updatePheromone(schedules, makespans)
            makespan_progress.append(bestMakespan)  # store the progress of make span each ittration
            #calculate average load for this iteration
            avgLoad = np.mean([sum(machine.totalProcessingTime for machine in schedule) for schedule in schedules])

            avgLoadPerMachine = []
            for machineId in range(self.machines1):
                #sum of the load for this machine
                sumLoadForMachine = sum(schedule[machineId].totalProcessingTime for schedule in schedules)
                #divide by the number of schedules for average load
                avgLoadPerMachine.append(sumLoadForMachine / len(schedules))

        #calculate average makespan
        avgMakespan = np.mean(makespans)

        # Calculate average load (average per machine)
        avgLoad = np.mean([sum(machine.totalProcessingTime for machine in schedule) for schedule in schedules])

        solutionQuality = self.calculateSolutionQuality(bestMakespan, worstMakespan, avgMakespan)
        return bestSchedule, bestMakespan, worstSchedule, worstMakespan, bestLoad, worstLoad, avgMakespan, avgLoad, avgLoadPerMachine, solutionQuality, makespan_progress


    #calculate solution quality
    def calculateSolutionQuality(self, bestMakespan, worstMakespan, makespan):
        if worstMakespan == bestMakespan:
            return 1.0  #if worst and best are equal, solution quality is perfect
        else:
            #calculate solution quality as a fraction between best and worst makespan
            return 1 - ((makespan - bestMakespan) / (worstMakespan - bestMakespan))


    #construct a schedule
    def createSchedule(self):
        machines2 = [Machine(i) for i in range(self.machines1)]  #create machine objects
        taskIds = list(range(self.tasks))  #list of task ids
        random.shuffle(taskIds)  #random task order

        for taskId in taskIds:
            machine = self.selectMachineForTask(machines2, taskId)

            try:
                processingTime = self.taskTimes[(taskId + 1, machine.machineId + 1)]  #Adjust indices
            except KeyError:
                print(f"KeyError: ({taskId + 1}, {machine.machineId + 1}) not found in taskTimes")
                continue

            machine.assignTask(Task(taskId), processingTime)

        return machines2

    #select a machine for the task
    def selectMachineForTask(self, machines, taskId):
        probabilities = [self.calculateProbability(taskId, machine) for machine in machines]
        totalProbability = sum(probabilities)
        probabilities = [p / totalProbability for p in probabilities]

        selectedMachine = np.random.choice(machines, p=probabilities)
        return selectedMachine

    # calculate the probability for a machine
    def calculateProbability(self, taskId, machine):
        pheromoneLevel = self.pheromoneMatrix[taskId][machine.machineId]  #pheromone level for this task machine pair
        heuristic = 1 / (machine.totalProcessingTime + 1)  #heuristic based on the machine's current load
        return (pheromoneLevel ** self.alpha) * (heuristic ** self.beta)

    #calculate the makespan of the schedule
    def calculateMakespan(self, schedule):
        return max(machine.totalProcessingTime for machine in schedule)

    #update the pheromone matrix based on the schedules found by the ants
    def updatePheromone(self, schedules, makespans):
        self.pheromoneMatrix *= (1 - self.pheromoneEvaporation)  #apply evaporation

        for schedule, makespan in zip(schedules, makespans):
            for machine in schedule:
                for task in machine.assignedTasks:
                    self.pheromoneMatrix[task.taskId][machine.machineId] += 1 / makespan  #reinforce good solutions


#reading and parsing benchmark data from a file
def readBenchmark(filePath):
    with open(filePath, 'r') as file:
        #read task and machine IDs
        taskIds = list(map(int, file.readline().split()))  #task IDs from the first line
        machineIds = list(map(int, file.readline().split()))  #machine IDs from the second line
        processingTimes = {}

        for line in file:
            line = line.strip()
            #ensure each line matches the expected format and contains three values
            if line and line.startswith('(') and line.endswith(')'):
                try:
                    taskId, machineId, time = line.strip('()').split(',')
                    taskId, machineId, time = int(taskId), int(machineId), int(time)
                    processingTimes[(taskId, machineId)] = time
                except ValueError as e:
                    print(f"Skipping invalid line: {line} due to error: {e}")

    return taskIds, machineIds, processingTimes

#charts
def plotGanttChart(schedule):  # Gantt Chart
    plt.figure(figsize=(10, 5))
    colors = plt.cm.get_cmap("tab10", len(schedule))
    for machine in schedule:
        start_time = 0
        for task in machine.assignedTasks:
            plt.barh(machine.machineId, task.processingTime, left=start_time, color=colors(machine.machineId), edgecolor='black')
            start_time += task.processingTime
    plt.xlabel("Time")
    plt.ylabel("Machines")
    plt.title("Gantt Chart for Task Scheduling")
    plt.show()

def plotMakespanProgress(makespan_progress):  # makespan progress
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(makespan_progress) + 1), makespan_progress, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Best Makespan")
    plt.title("Makespan Progress over Iterations")
    plt.grid(True)
    plt.show()

def plotMachineLoad(schedule):  #load analysis
    machine_loads = [machine.totalProcessingTime for machine in schedule]
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(machine_loads) + 1), machine_loads, color='skyblue', edgecolor='black')
    plt.xlabel("Machine")
    plt.ylabel("Load")
    plt.title("Machine Load Distribution")
    plt.show()

def main():
    filePath = "/benchmark.txt"
    taskIds, machineIds, taskTimes = readBenchmark(filePath)
    tasks, machines1 = len(taskIds), len(machineIds)

    aco = TaskSchedulingACO(
        ants=10,
        tasks=tasks,
        machines1=machines1,
        taskTimes=taskTimes
    )

    bestSchedule, bestMakespan, worstSchedule, worstMakespan, bestLoad, worstLoad, avgMakespan, avgLoad, avgLoadPerMachine, solutionQuality, makespan_progress = aco.findOptimalSchedule(maxIterations=100)

    #printing best Schedule
    print("The best schedule:")
    for machine in bestSchedule:
        print(f"machine no. {machine.machineId + 1}: " +
              f"{[f'task: {task.taskId + 1}, with processing time: {task.processingTime}' for task in machine.assignedTasks]} " +
              f"(total time calculated: {machine.totalProcessingTime})")
    print(f"best case makespan: {bestMakespan}")
    print(f"best load: {bestLoad}")

    #printing worst Case
    print("The worst case:")
    for machine in worstSchedule:
        print(f"machine no. {machine.machineId + 1}: " +
              f"{[f'task: {task.taskId + 1}, with processing time: {task.processingTime}' for task in machine.assignedTasks]} " +
              f"(total time calculated: {machine.totalProcessingTime})")
    print(f"worst case makespan: {worstMakespan}")
    print(f"worst load: {worstLoad}")

    #printing average
    print(f"average makespan: {avgMakespan}")
    print(f"average load across all machines: {avgLoad}")
    print(f"average load for each machine: {avgLoadPerMachine}")
    print(f"Solution Quality: {solutionQuality}")

    # Visualization
    plotGanttChart(bestSchedule)
    plotMakespanProgress(makespan_progress)
    plotMachineLoad(bestSchedule)


if __name__ == "__main__":
    main()
