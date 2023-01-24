import numpy as np
import pandas as pd
from scipy import stats as st
import time
import itertools


class Bet:

    min_bet = 10
    max_bet = 5000

    def __init__(self, location, amount = 0):
        self.location = location
        self.amount = amount

    def change_bet(self, location, amount):
        self.amount += amount
        if amount > 0:
            print(f"Placed ${amount} on {location}")
        else:
            print(f"Removed ${np.abs(amount)} from {location}")
        print(f"Total bet: {self.amount}")


class Player:

    def __init__(self, name, balance = 0, stoploss = 0):
        self.name = name
        self.balance = balance
        self.stoploss = stoploss
        self.bets = np.array([0,0,0,0,0])

    def winlose(self, amount):
        self.balance += amount

    def make_bet(self, amount):
        self.bets = np.append(self.bets, amount, axis=0)

class Roll:

    def __init__(self, die1, die2):
        self.die1 = die1
        self.die2 = die2 

    def dice_sum(self, die1, die2):
        self.dice_sum = die1 + die2

    def come_out(self):
        pass


start_time = time.time()
#Table data
num_players = 8

#Generate dice rolls
num_rolls = 1000
seed_seed = np.random.randint(0, 100000)
seed = np.random.seed(seed_seed)
die_sides = [1, 2, 3, 4, 5, 6]
die_1 = np.random.choice(die_sides, num_rolls)
die_2 = np.random.choice(die_sides, num_rolls)
dice_sum = die_1 + die_2


#Determine table status
comeout = np.ones(num_rolls, dtype=int)
shooter = np.ones(num_rolls, dtype=int)
point = np.zeros(num_rolls, dtype=int)
current_shooter = 0

#Determine whether roll is come out roll, set point
for roll in range(0, num_rolls - 1):
    if (comeout[roll] == 1):
        if dice_sum[roll] in {7, 11}:
            comeout[roll + 1] = 1
        elif dice_sum[roll] in {2, 3, 12}:
            point[roll + 1] == 0
        else:
            comeout[roll + 1] = 0
            point[roll + 1] = dice_sum[roll]
    else:
        if (dice_sum[roll] == 7):
            current_shooter += 1
            comeout[roll + 1] = 1
        elif point[roll] == dice_sum[roll]:
            point[roll + 1] = 0
            comeout[roll + 1] = 1
        else:
            point[roll + 1] = point[roll]
            comeout[roll + 1] = 0
    shooter[roll + 1] = np.mod(current_shooter, num_players) + 1

win_pass = np.zeros(num_rolls, dtype=int)
win_odds_pass = np.zeros(num_rolls, dtype=int)
    
#Determine whether pass bets win
for roll in range(0, num_rolls):
    if (comeout[roll] == 1):
        if dice_sum[roll] in {7, 11}:
            win_pass[roll] = 1
        elif dice_sum[roll] in {2, 3, 12}:
            win_pass[roll] = -1
    elif (comeout[roll] == 0):
        if dice_sum[roll] == point[roll]:
            win_pass[roll] = 1
            win_odds_pass[roll] = 1
        elif dice_sum[roll] == 7:
            win_pass[roll] = -1
            win_odds_pass[roll] = -1

win_come = np.zeros(num_rolls, dtype=int)
win_odds_come = np.zeros((6, num_rolls), dtype=int)

#Determine whether come bets win
for roll in range(0, num_rolls):
    if (comeout[roll] == 0):
        if dice_sum[roll] in {7, 11}:
            win_come[roll] = 1
        elif dice_sum[roll] in {2, 3, 12}:
            win_come[roll] = -1
        else:
            win_come[roll] = 0

win_came = np.zeros((6, num_rolls), dtype=int)
win_odds_came = win_came.copy()
come_tracker = np.zeros((6, 1), dtype = int)

#Determine whether come bets win after moving to numbers
for roll in range(0, num_rolls):
    if dice_sum[roll] in {4, 5, 6}:
        if come_tracker[dice_sum[roll] - 4] == 1:
            win_came[dice_sum[roll] - 4, roll] = 1
        if comeout[roll] == 0:
            come_tracker[dice_sum[roll] - 4] = 1
    elif dice_sum[roll] in {8, 9, 10}:
        if come_tracker[dice_sum[roll] - 5] == 1:
            win_came[dice_sum[roll] - 5, roll] = 1
        if comeout[roll] == 0:
            come_tracker[dice_sum[roll] - 5] = 1
    elif dice_sum[roll] == 7:
        win_came[:, [roll]] = np.negative(come_tracker)
        come_tracker.fill(0)

win_place = np.zeros((6, num_rolls), dtype=int)
win_lay = np.zeros((6, num_rolls), dtype=int)

#Determine whether place, buy, lay bets win
for roll in range(0, num_rolls):
    if dice_sum[roll] in {4, 5, 6}:
        win_lay[dice_sum[roll] - 4, roll] = -1
    elif dice_sum[roll] in {8, 9, 10}:
        win_lay[dice_sum[roll] - 5, roll] = -1
    elif dice_sum[roll] == 7:
        win_lay[:, [roll]] = 1
    if (comeout[roll] == 0):
        if dice_sum[roll] in {4, 5, 6}:
            win_place[dice_sum[roll] - 4, roll] = 1
        elif dice_sum[roll] in {8, 9, 10}:
            win_place[dice_sum[roll] - 5, roll] = 1
        elif dice_sum[roll] == 7:
            win_place[:, roll] = -1

win_buy = win_place.copy()

win_field = np.zeros((6, num_rolls), dtype=int) - 1
win_any_craps = np.zeros((num_rolls), dtype=int) - 1
win_3_or_11 = np.zeros((num_rolls), dtype=int) - 1
win_2_or_12 = np.zeros((num_rolls), dtype=int) - 1

#Determine whether range bets win
for roll in range(0, num_rolls):
    if dice_sum[roll] in {2, 3, 4, 9, 10, 11, 12}:
        win_field = 1
    if dice_sum[roll] in {2, 3, 12}:
        win_any_craps[roll] = 1
    if dice_sum[roll] in {3, 11}:
        win_3_or_11[roll] = 1
    if dice_sum[roll] in {2, 12}:
        win_2_or_12[roll] = 1

win_big6 = np.zeros(num_rolls, dtype=int)
win_big8 = np.zeros(num_rolls, dtype=int)

#Determine whether standing single numbers win
for roll in range(0, num_rolls):
    if (comeout[roll] == 0):
        if dice_sum[roll] == 6:
            win_big6[roll] = 1
        elif dice_sum[roll] == 8:
            win_big8[roll] = 1
        elif dice_sum[roll] == 7:
            win_big6[roll] = -1
            win_big8[roll] = -1

win_hardway = np.zeros((4, num_rolls), dtype=int)

#Determine whether hardways win
for roll in range(0, num_rolls):
    if (comeout[roll] == 0):
        if dice_sum[roll] in {4, 6, 8, 10}:
            if die_1[roll] == die_2[roll]:
                win_hardway[die_1[roll] - 2, roll] = 1
            else:
                win_hardway[int(dice_sum[roll] / 2 - 2), roll] = -1
        elif dice_sum[roll] == 7:
            win_hardway[:, roll] = -1

win_any_7 = np.zeros(num_rolls, dtype=int) - 1
win_c_and_e = np.zeros((2, num_rolls), dtype=int) - 1

#Determine whether one-roll single numbers win
for roll in range(0, num_rolls):
    if dice_sum[roll] == 7:
        win_any_7[roll] = 1
    if dice_sum[roll] == 11:
        win_c_and_e[1, roll] = 1
    
win_c_and_e[0] = win_any_7

win_dont_pass = -win_pass.copy()
win_dont_come = -win_come.copy()
#win_dont_came = 
win_odds_dont_pass = -win_odds_pass.copy()
win_odds_dont_come = -win_odds_come.copy()
#win_odds_dont_came = 

bets_dict = dict({"pass_line": 1, "dont_pass": 1, "come": 1, "dont_come": 1,
                  "place4": 9/5, "place5": 7/5, "place6": 7/6, "place8": 7/6, "place9": 7/5, "place10": 9/5,
                  "buy4": 2, "buy5": 3/2, "buy6": 6/5, "buy8": 6/5, "buy9": 3/2, "buy10": 2,
                  "lay4": 1/2, "lay5": 2/3, "lay6": 5/6, "lay8": 5/6, "lay9": 2/3, "lay10": 1/2,
                  "odds_pass4": 2, "odds_pass5": 3/2, "odds_pass6": 6/5, "odds_pass8": 6/5, "odds_pass9": 3/2, "odds_pass10": 2,
                  "odds_dont_pass4": 1, "odds_dont_pass5": 1, "odds_dont_pass6": 1, "odds_dont_pass8": 1, "odds_dont_pass9": 1, "odds_dont_pass10": 1,
                  "odds_come4": 2, "odds_come5": 3/2, "odds_come6": 6/5, "odds_come8": 6/5, "odds_come9": 3/2, "odds_come10": 2,
                  "odds_dont_come4": 1, "odds_dont_come5": 1, "odds_dont_come6": 1, "odds_dont_come8": 1, "odds_dont_come9": 1, "odds_dont_come10": 1,
                  "field2": 2, "field3": 1, "field4": 1, "field9": 1, "field10": 1, "field11": 1, "field12": 12,
                  "prop_any7": 4, "prop_any_craps": 7, "prop_hard4": 7, "prop_hard6": 9, "prop_hard8": 9, "prop_hard10": 7,
                  "horn2": 6.75, "horn3": 3, "horn11": 3, "horn12": 6.75,
                  "horn_high2": 26, "horn_high3": 11, "horn_high11": 11, "horn_high12": 26,
                  "craps2": 30, "craps3": 15, "craps11": 15, "craps12": 30,
                  "hop22": 30, "hop33": 30, "hop44": 30, "hop55": 30, "hop13": 15, "hop14": 15, "hop15": 15, "hop16": 15, "hop23": 15, "hop24": 15, "hop25": 15, "hop26": 15, "hop34": 15, "hop35": 15, "hop36": 15, "hop45": 15, "hop46": 15})

actual_odds_dict = dict({"pass_line": 244/495, "dont_pass": 2847/5940, "come": 244/495, "dont_come": 2847/5940,
                    })

bets_list = list(bets_dict.keys())
odds_list = list(bets_dict.values())


player_bets = ['pass_line']
player_wagers = [10]
player_odds = bets_dict[player_bets[0]]


#Create a summary DataFrame
player_win_pass = win_pass * player_wagers
game_data_titles = ['']
game_data = np.zeros((4,11))
game_summary = pd.DataFrame(game_data, columns = ['Bet', 'Win Chance', 'House Edge', 'Active %', 'Amount Won', 'Wager', 'Total Wagered', 'Rolls Total', 'Rolls Won', 'Rolls Lost', 'Rolls Pushed'])
game_summary['Bet'] = bets_list[:4]
game_summary['Win Chance'] = actual_odds_dict.values()

print(player_bets[0], " Amount won: $", np.sum(player_win_pass), "Wins:", sum(1 for i in player_win_pass if i > 0), "Losses:", sum(1 for i in player_win_pass if i < 0))
print(game_summary)


"""

test_name = "Come Bet"
tested_results = win_come                       #which bet to analyze(one row)
actual_odds = 8/36  #actual_odds_dict['pass_line']     #actual odds of that bet
result_to_check = 8/35                       #determine chance of specific result

start_time2 = time.time()
print(np.count_nonzero(tested_results == -1))
print(np.count_nonzero(tested_results == 1))



#pass line real stats
s1 = np.sqrt((actual_odds * (1 - actual_odds))/num_rolls)      #standard deviation
Z1 = result_to_check - actual_odds       #observed - expected       eg. standard normal deviate
z1 = Z1 / s1       #z-score
p1 = st.norm.sf(abs(z1))

#pass line game stats
x = np.bincount(tested_results + 1) #bincount is the fastest wayof calculating occurrences
s2 = np.sqrt((actual_odds * (1 - actual_odds))/num_rolls)
Z2 = x[2]/(x[0] + x[2]) - actual_odds
z2 = Z2 / s2
p2 = st.norm.sf(abs(z2))

#Test accuracy of my own equations
sample_size = 100
num_samples = 1000
results = np.zeros(num_samples)
for i in range(0, num_samples):
    sample = np.random.choice(win_pass + 1, size = sample_size, replace = False)
    x3 = np.bincount(sample)
    results[i] = x3[2]/(x3[0] + x3[2])
test_avg = np.mean(results)
test_std = np.std(results)
s3 = np.sqrt((actual_odds * (1 - actual_odds))/num_samples)
Z3 = test_avg - actual_odds
PE = (actual_odds - test_avg) / (actual_odds) * 100
z3 = Z3 / s3
p3 = st.norm.sf(abs(z3))

print(test_name, ":", " Losses:", x[0], " Push:", x[1], " Wins:", x[2], " Win %:", round(x[2]/(x[0]+x[2])*100, 2), " vs ", round(actual_odds*100, 2))
print("SD:", round(s1, 3), "SND:", round(Z1, 4), "z:", round(z1, 4), " p:", round(p1*100, 2), "%", " 1 in ", round(1/p1))
print("SD:", round(s2, 3), "SND:", round(Z2, 4), "z:", round(z2, 4), " p:", round(p2*100, 2))#, "%", " 1 in ", round(1/p2))
print("SD:", round(s3, 3), "SND:", round(PE, 4), "z:", round(z3, 4), " p:", round(p3*100, 2))#, "%", " 1 in ", round(1/p3))





win_pass                ALL GOOD
win_odds_pass           
win_dont_pass
win_odds_dont_pass
win_come                ALL GOOD
win_odds_come
win_dont_come
win_odds_dont_come
win_came
win_odds_came
#win_dont_came = 
#win_odds_dont_came = 
win_place
win_lay
win_buy
win_field
win_any_craps
win_3_or_11
win_2_or_12
win_big6
win_big8
win_hardway
win_any_7
win_c_and_e



print(np.stack((dice_sum, point, comeout)))

#print(np.stack((die_1, die_2, dice_sum, point, comeout, shooter)), seed)

#Summary statistics
unique, counts = np.unique(dice_sum, return_counts=True)
dice_sum_stats = dict(zip(unique, counts))
print("Dice rolls:", dice_sum_stats)
"""
print("Run time: ", f'{(time.time() - start_time)*1000:.3f}', "ms", seed_seed)

