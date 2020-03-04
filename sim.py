#!/bin/python3

from pid import PIDArduino
from autotune import PIDAutotune
from kettle import Kettle
from collections import deque, namedtuple
import sys
import math
import logging
import argparse
import matplotlib.pyplot as plt
from Adam import AdamOptimiser


LOG_FORMAT = '%(name)s: %(message)s'
Simulation = namedtuple(
    'Simulation',
    ['name', 'sut', 'kettle', 'delayed_temps', 'timestamps',
     'heater_temps', 'sensor_temps', 'outputs'])
PID_dict = {'No PID':[1,0,0]}
PID_finetune = {}
PID_names = ['Kp','Ki','Kd']


'''
Parameter management
'''
def parser_add_args(parser):
    parser.add_argument(
        '-p', '--pid', dest='pid', nargs=4, metavar=('name', 'kp', 'ki', 'kd'),
        default=None, action='append', help='simulate a PID controller')
    parser.add_argument(
        '-a', '--atune', dest='autotune', default=False,
        action='store_true', help='simulate autotune')

    parser.add_argument(
        '-v', '--verbose', dest='verbose', default=0,
        action='count', help='be verbose')
    parser.add_argument(
        '-e', '--export', dest='export', default=False,
        action='store_true', help='export data to a .csv file')
    parser.add_argument(
        '-n', '--noplot', dest='noplot', default=False,
        action='store_true', help='do not plot the results')

    parser.add_argument(
        '-t', '--temp', dest='kettle_temp', metavar='T', default=40.0,
        type=float, help='initial kettle temperature in °C (default: 40)')
    parser.add_argument(
        '-s', '--setpoint', dest='setpoint', metavar='T', default=45.0,
        type=float, help='target temperature in °C (default: 45)')
    parser.add_argument(
        '--ambient', dest='ambient_temp', metavar='T', default=20.0,
        type=float, help='ambient temperature in °C (default: 20)')

    parser.add_argument(
        '-i', '--interval', dest='interval', metavar='t', default=20,
        type=int, help='simulated interval in minutes (default: 20)')
    parser.add_argument(
        '-d', '--delay', dest='delay', metavar='t', default=15.0,
        type=float, help='system response delay in seconds (default: 15)')
    parser.add_argument(
        '--sampletime', dest='sampletime', metavar='t', default=5.0,
        type=float, help='temperature sample time in seconds (default: 5)')

    parser.add_argument(
        '--volume', dest='volume', metavar='V', default=70.0,
        type=float, help='kettle content volume in liters (default: 70)')
    parser.add_argument(
        '--diameter', dest='diameter', metavar='d', default=50.0,
        type=float, help='kettle diameter in cm (default: 50)')

    parser.add_argument(
        '--power', dest='heater_power', metavar='P', default=6.0,
        type=float, help='heater power in kW (default: 6)')
    parser.add_argument(
        '--heatloss', dest='heat_loss_factor', metavar='x', default=1.0,
        type=float, help='kettle heat loss factor (default: 1)')
    parser.add_argument(
        '--minout', dest='out_min', metavar='x', default=0.0,
        type=float, help='minimum PID controller output (default: 0)')
    parser.add_argument(
        '--maxout', dest='out_max', metavar='x', default=100.0,
        type=float, help='maximum PID controller output (default: 100)')
    parser.add_argument(
        '--delta_pid', dest='delta_pid', nargs=3,
        default=[0.1,0.0005,1], action='append', help='Delta kp,ki,kd for calculating gradient')
    parser.add_argument(
        '--lr', dest='lr', nargs=3,
        default=[2,0.001,5], action='append', help='Learning rate for kp ki kd')
    parser.add_argument(
        '--max_steps', dest='max_steps', default=1000,
        type=float, help='Max steps for tuning (default: 500)')
    parser.add_argument(
        '--time constant', dest='time_constant', default=75,
        type=float, help='Time constant, we will start calculating error after that time(default: 100)')

    

def write_csv(sim):
    filename = sim.name + '.csv'
    with open(filename, 'w+') as csv:
        csv.write('timestamp;output;sensor_temp;heater_temp\n')

        for i in range(0, len(sim.timestamps)):
            csv.write('{0};{1:.2f};{2:.2f};{3:.2f}\n'.format(
                sim.timestamps[i], sim.outputs[i], sim.sensor_temps[i], sim.heater_temps[i]))
            
            
'''
Update simulation's condition
'''
def sim_update(sim, timestamp, output, args):
    sim.kettle.heat(args.heater_power * (output / 100), args.sampletime)
    sim.kettle.cool(args.sampletime, args.ambient_temp, args.heat_loss_factor)
    sim.delayed_temps.append(sim.kettle.temperature)
    sim.timestamps.append(timestamp)
    sim.outputs.append(output)
    sim.sensor_temps.append(sim.delayed_temps[0])
    sim.heater_temps.append(sim.kettle.temperature)



'''
Plot figure
'''

def plot_simulations(simulations, title):
    lines = []
    fig, ax1 = plt.subplots()
    upper_limit = 0

    # Try to limit the y-axis to a more relevant area if possible
    for sim in simulations:
        m = max(sim.sensor_temps) + 1
        upper_limit = max(upper_limit, m)

    if upper_limit > args.setpoint:
        lower_limit = args.setpoint - (upper_limit - args.setpoint)
        ax1.set_ylim(lower_limit, upper_limit)

    # Create x-axis and first y-axis (temperature)
    ax1.plot()
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('temperature (°C)')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # Draw setpoint line
    lines += [plt.axhline(
        y=args.setpoint, color='r', linestyle=':', linewidth=0.9, label='setpoint')]

    # Create second y-axis (power)
    ax2 = ax1.twinx()
    ax2.set_ylabel('power (%)')

    # Plot temperature and output values
    i = 0
    for sim in simulations:
        color_cycle_idx = 'C' + str(i)
        lines += ax1.plot(
            sim.timestamps, sim.sensor_temps, color=color_cycle_idx,
            alpha=1.0, label='{0}: temp.'.format(sim.name))
        lines += ax2.plot(
            sim.timestamps, sim.outputs, '--', color=color_cycle_idx,
            linewidth=1, alpha=0.7, label='{0}: output'.format(sim.name))
        i += 1

    # Create legend
    labels = [l.get_label() for l in lines]
    offset = math.ceil((1 + len(simulations) * 2) / 3) * 0.05
    ax1.legend(lines, labels, loc=9, bbox_to_anchor=(
        0.5, -0.1 - offset), ncol=3)
    fig.subplots_adjust(bottom=0.2 + offset)

    # Set title
    plt.title(title)
    fig.canvas.set_window_title(title)
    plt.show()

'''
Auto tuning with relay based method
'''
def simulate_autotune(args):
    timestamp = 0  # seconds
    maxlen = max(1, round(args.delay / args.sampletime))
    delayed_temps = deque(maxlen=maxlen)
    delayed_temps.extend(maxlen * [args.kettle_temp])

#    Simulation = namedtuple(
#        'Simulation',
#        ['name', 'sut', 'kettle', 'delayed_temps', 'timestamps',
#         'heater_temps', 'sensor_temps', 'outputs'])
    
    sim = Simulation(
        'autotune',
        PIDAutotune(
            args.setpoint, 100, args.sampletime, out_min=args.out_min,
            out_max=args.out_max, time=lambda: timestamp),
        Kettle(args.diameter, args.volume, args.kettle_temp),
        delayed_temps,
        [], [], [], []
    )

    # Run autotune until completed
    while not sim.sut.run(sim.delayed_temps[0]):
        timestamp += args.sampletime
        sim_update(sim, timestamp, sim.sut.output, args)
        if args.verbose > 0:
                print('time:    {0} sec'.format(timestamp))
                print('state: {0}'.format(sim.sut.state))
                print('{0}: {1:.2f}%'.format(sim.name, sim.sut.output))
                print('temp sensor:    {0:.2f}°C'.format(sim.sensor_temps[-1]))
                print('temp heater:    {0:.2f}°C'.format(sim.heater_temps[-1]))
                print()


    print('time:    {0} min'.format(round(timestamp / 60)))
    print('state:   {0}'.format(sim.sut.state))
    print()
    

    # On success, print params for each tuning rule
    if sim.sut.state == PIDAutotune.STATE_SUCCEEDED:
        for rule in sim.sut.tuning_rules:
            params = sim.sut.get_pid_parameters(rule)
            PID_dict[rule] = [params.Kp,params.Ki,params.Kd]
            if rule == "ziegler-nichols":
                PID_finetune[rule] =  [params.Kp,params.Ki,params.Kd]
            print('Rule: {0}'.format(rule))
            print('Kp: {0}'.format(params.Kp))
            print('Ki: {0}'.format(params.Ki))
            print('Kd: {0}'.format(params.Kd))
            print()


    if args.export:
        write_csv(sim)

    if not args.noplot:
        title = 'PID autotune, {0:.1f}l kettle, {1:.1f}kW heater, {2:.1f}s delay'.format(
            args.volume, args.heater_power, args.delay)
        plot_simulations([sim], title)

'''
PID simulation and plot

input: 
    args:parameters management
    PID_dict: a PID dictory {rule:[Kp,Ki,Kd]}
output:
    a figure 
'''
def simulate_pid(args,PID_dict):
    timestamp = 0  # seconds
    delayed_temps_len = max(1, round(args.delay / args.sampletime))
    sims = []

    # Create a simulation for each tuple pid(name, kp, ki, kd)
    for key,pid in PID_dict.items():
        sim = Simulation(
            key,
            PIDArduino(
                args.sampletime, float(pid[0]), float(pid[1]), float(pid[2]),
                args.out_min, args.out_max, lambda: timestamp),
            Kettle(args.diameter, args.volume, args.kettle_temp),
            deque(maxlen=delayed_temps_len),
            [], [], [], []
        )
        sims.append(sim)

    # Init delayed_temps deque for each simulation
    for sim in sims:
        sim.delayed_temps.extend(sim.delayed_temps.maxlen * [args.kettle_temp])

    # Run simulation for specified interval
    while timestamp < (args.interval * 60):
        timestamp += args.sampletime

        for sim in sims:
            output = sim.sut.calc(sim.delayed_temps[0], args.setpoint)
            output = max(output, 0)
            output = min(output, 100)
            sim_update(sim, timestamp, output, args)

            if args.verbose > 0:
                print('time:    {0} sec'.format(timestamp))
                print('{0}: {1:.2f}%'.format(sim.name, output))
                print('temp sensor:    {0:.2f}°C'.format(sim.sensor_temps[-1]))
                print('temp heater:    {0:.2f}°C'.format(sim.heater_temps[-1]))
        if args.verbose > 0:
            print()

    if args.export:
        for sim in sims:
            write_csv(sim)

    if not args.noplot:
        title = 'PID simulation, {0:.1f}l kettle, {1:.1f}kW heater, {2:.1f}s delay'.format(
            args.volume, args.heater_power, args.delay)
        plot_simulations(sims, title)
        
        
'''
caculate cost function
input:
    tuning_rule: name of tuning rule 
    pid_pre_tune: pid parameters calculated with relay method
output:
    J: cost function which need to be minimized
''' 
    
def calc_cost(tuning_rule,pid_pre_tune):
    timestamp = 0
    cost = 0
    n_err = 0
    delayed_temps_len = max(1, round(args.delay / args.sampletime))
    sim = Simulation(
            tuning_rule,
            PIDArduino(
                args.sampletime, float(pid_pre_tune[0]), float(pid_pre_tune[1]), float(pid_pre_tune[2]),
                args.out_min, args.out_max, lambda: timestamp),
            Kettle(args.diameter, args.volume, args.kettle_temp),
            deque(maxlen=delayed_temps_len),
            [], [], [], []
        )
    sim.delayed_temps.extend(sim.delayed_temps.maxlen * [args.kettle_temp])
    # run system for a specified interval
    while timestamp < (args.interval * 60):
        timestamp += args.sampletime
        output = sim.sut.calc(sim.delayed_temps[0], args.setpoint)
        output = max(output, 0)
        output = min(output, 100)
        sim_update(sim, timestamp, output, args)
        #start calculating error and cost(mse)
        if timestamp > args.time_constant:
            n_err += 1
            err = args.setpoint-sim.sensor_temps[-1]
            cost += err**2
    J = cost/n_err
    return J

'''
calculate gradient for kp,ki,kd
input:
    tunning_rule: name of rule of pre-tuned pid
    pid_pre_tune: pid parameter with relay method
    index: index for kp,ki,kd (0,1,2)
output:
    grad:gradient = (J(w+1)-J(w-1))/(2)
'''

def calc_grad(tuning_rule,pid_pre_tune,index):
    pid_pre_tune[index] -= args.delta_pid[index]
    J_ = calc_cost(tuning_rule,pid_pre_tune)
    #calculate gradient dJ/dkp
    pid_pre_tune[index] += args.delta_pid[index]
    J = calc_cost(tuning_rule,pid_pre_tune)
    grad = (J-J_)/(2*args.delta_pid[index])
    return grad

'''
gradient descent fine tune
input:
    tuning rule: name of rule
    pid_pre_tune:pid parameter with relay method
output:
    PID_finetune: pid parameter fine tuned
'''
def GD_finetune(tuning_rule,pid_pre_tune):
    for step in range(args.max_steps):
        for index in range(3):
            grad = calc_grad(tuning_rule,pid_pre_tune,index)
            pid_pre_tune[index] = max(pid_pre_tune[index] - args.lr[index]*grad,0)
    for index in range(3):
        print('Fine tune {} finished:'.format(PID_names[index]))
        print('{}:{}'.format(PID_names[index],pid_pre_tune[index]))
    
    PID_finetune['Fine tune'] = pid_pre_tune
    return PID_finetune
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser_add_args(parser)

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()

        if args.verbose > 1:
            logging.basicConfig(stream=sys.stderr, format=LOG_FORMAT, level=logging.DEBUG)
        if args.autotune:
            simulate_autotune(args)
            simulate_pid(args,PID_dict)
            pid_pretune = PID_dict["ziegler-nichols"]
            simulate_pid(args,GD_finetune("ziegler-nichols",pid_pretune))
            