import csv
csv_columns = ['Model','Epochs','KL', 'Total Variance Distance', 'JS', 'FID', 'IS']
train_rows = [
{'Model': 'Small', 'Epochs': 0, 'KL': -8.3612, 'Total Variance Distance': 0.4999, 'JS':-3.4877, 'FID':375.703595647013, 'IS':1.2427706785749826},
{'Model': 'Small', 'Epochs': 25, 'KL': 4.7701, 'Total Variance Distance': 130.1956, 'JS':-1.7090, 'FID':136.18441423770338, 'IS':1.8208543390803444},
{'Model': 'Small', 'Epochs': 50, 'KL': 3.9159, 'Total Variance Distance': 38.7184, 'JS':-1.2949, 'FID':107.18872122222791, 'IS':2.1164940184481886},
{'Model': 'Small', 'Epochs': 75, 'KL': 3.5472, 'Total Variance Distance': 24.1821, 'JS':-1.1202, 'FID':95.67289993841712, 'IS':2.3021846205780316},
{'Model': 'Small', 'Epochs': 100, 'KL': 3.2245, 'Total Variance Distance': 16.1876, 'JS':-0.9704, 'FID':89.97673588683574, 'IS':2.429979131759416},
{'Model': 'Small', 'Epochs': 125, 'KL': 2.8571, 'Total Variance Distance': 10.6201, 'JS':-0.8052, 'FID':90.37035669217863, 'IS':2.4083579561346657},
{'Model': 'Small', 'Epochs': 150, 'KL': 2.5460, 'Total Variance Distance': 7.6072, 'JS':-0.6716, 'FID':93.35474143895999, 'IS':2.3816469392706368},
{'Model': 'Small', 'Epochs': 175, 'KL': 2.3392, 'Total Variance Distance': 6.4052, 'JS':-0.5880, 'FID':94.78436847726852, 'IS':2.4010978757710775},
{'Model': 'Small', 'Epochs': 200, 'KL': 2.2377, 'Total Variance Distance': 7.9057, 'JS':-0.5505, 'FID':95.67801522152837, 'IS':2.3622274043457097},
{'Model': 'Small', 'Epochs': 225, 'KL': 2.2585, 'Total Variance Distance': 32.0204, 'JS':-0.5658, 'FID':97.69469469425235, 'IS':2.3862274570377493},
{'Model': 'Small', 'Epochs': 250, 'KL': 2.2828, 'Total Variance Distance': 332.9601, 'JS':-0.5850, 'FID':99.63915625798319, 'IS':2.363172870000154},

{'Model': 'Medium', 'Epochs': 0, 'KL': -8.3241, 'Total Variance Distance': 0.4998, 'JS':-3.4692, 'FID':376.6236941265725, 'IS':1.2348173478412139},
{'Model': 'Medium', 'Epochs': 25, 'KL': 4.3116, 'Total Variance Distance': 69.6932, 'JS':-1.4848, 'FID':118.21016364587604, 'IS':2.0008240688736696},
{'Model': 'Medium', 'Epochs': 50, 'KL': 3.5874, 'Total Variance Distance': 23.9211, 'JS':-1.1377, 'FID':93.13635648600643, 'IS':2.284198851706024},
{'Model': 'Medium', 'Epochs': 75, 'KL': 3.3046, 'Total Variance Distance': 17.2887, 'JS':-1.0064, 'FID':82.61445581224956, 'IS':2.44977795384239},
{'Model': 'Medium', 'Epochs': 100, 'KL': 2.9235, 'Total Variance Distance': 11.3209, 'JS':-0.8345, 'FID':78.95948608339882, 'IS':2.5196854840362928},
{'Model': 'Medium', 'Epochs': 125, 'KL': 2.4046, 'Total Variance Distance': 6.6010, 'JS':-0.6158, 'FID':80.85456892671675, 'IS':2.507172374461161},
{'Model': 'Medium', 'Epochs': 150, 'KL': 1.8802, 'Total Variance Distance': 3.7648, 'JS':-0.4171, 'FID':82.50277966485515, 'IS':2.5233912142121553},
{'Model': 'Medium', 'Epochs': 175, 'KL': 1.4539, 'Total Variance Distance': 2.3158, 'JS':-0.2756, 'FID':85.04552533934293, 'IS':2.4557898208970497},
{'Model': 'Medium', 'Epochs': 200, 'KL': 1.1074, 'Total Variance Distance': 1.9177, 'JS':-0.1790, 'FID':87.789639391175, 'IS':2.4615996103643694},
{'Model': 'Medium', 'Epochs': 225, 'KL': 0.8335, 'Total Variance Distance': 4.2085, 'JS':-0.1188, 'FID':90.57263945118541, 'IS':2.4448607219946124},
{'Model': 'Medium', 'Epochs': 250, 'KL': 0.6468, 'Total Variance Distance': 4.0823, 'JS':-0.0875, 'FID':95.65387186837643, 'IS':2.391574085475964},

{'Model': 'Large', 'Epochs': 0, 'KL': -2.0747670017679902, 'Total Variance Distance': 0.4220510049830441, 'JS':-0.4848431215596609, 'FID':376.6750705733627, 'IS':1.236842138405124},
{'Model': 'Large', 'Epochs': 25, 'KL': 3.9279, 'Total Variance Distance': 37.1095, 'JS':-1.2994, 'FID':118.5399769679521, 'IS':1.9984233112416594},
{'Model': 'Large', 'Epochs': 50, 'KL': 3.4672, 'Total Variance Distance': 20.6168, 'JS':-1.0813, 'FID':89.86348283131366, 'IS':2.3211677423221966},
{'Model': 'Large', 'Epochs': 75, 'KL': 3.1881, 'Total Variance Distance': 14.9347, 'JS':-0.9525, 'FID':79.57145812239372, 'IS':2.507487040344116},
{'Model': 'Large', 'Epochs': 100, 'KL': 2.7829, 'Total Variance Distance': 9.6320, 'JS':-0.7729, 'FID':78.6458192134985, 'IS':2.501101304361945},
{'Model': 'Large', 'Epochs': 125, 'KL': 2.1320, 'Total Variance Distance': 4.9520, 'JS':-0.5101, 'FID':81.01922441951626, 'IS':2.476919766238947},
{'Model': 'Large', 'Epochs': 150, 'KL': 1.5287, 'Total Variance Distance': 2.5961, 'JS':-0.3022, 'FID':82.93932793127476, 'IS':2.4598211809621744},
{'Model': 'Large', 'Epochs': 175, 'KL': 1.0694, 'Total Variance Distance': 2135.3757, 'JS':-0.1743, 'FID':87.29892539976942, 'IS':2.4458626028917907},
{'Model': 'Large', 'Epochs': 200, 'KL': 0.6907, 'Total Variance Distance': 7.1126, 'JS':-0.0922, 'FID':94.2688239116095, 'IS':2.369513348128894},
{'Model': 'Large', 'Epochs': 225, 'KL': 0.2444, 'Total Variance Distance': 12.8788, 'JS':-0.0448, 'FID':102.37616265868098, 'IS':2.3054664581408617},
{'Model': 'Large', 'Epochs': 250, 'KL': 0.1028, 'Total Variance Distance': 30.8255, 'JS':-0.0411, 'FID':111.54278016725766, 'IS':2.2671213042415364},
]


test_rows = [
{'Model': 'Small', 'Epochs': 0, 'KL': -2.0726003503112045, 'Total Variance Distance': 0.4213412127953142, 'JS':-0.4846600401891214, 'FID':375.703595647013, 'IS':1.2427706785749826},
{'Model': 'Small', 'Epochs': 25, 'KL': 4.9084, 'Total Variance Distance': 169.8044, 'JS':-1.7768, 'FID':136.18441423770338, 'IS':1.8208543390803444},
{'Model': 'Small', 'Epochs': 50, 'KL': 4.0652, 'Total Variance Distance': 49.4468, 'JS':-1.3665, 'FID':107.18872122222791, 'IS':2.1164940184481886},
{'Model': 'Small', 'Epochs': 75, 'KL': 3.8047, 'Total Variance Distance': 35.5562, 'JS':-1.2421, 'FID':95.67289993841712, 'IS':2.3021846205780316},
{'Model': 'Small', 'Epochs': 100, 'KL': 3.8383, 'Total Variance Distance': 157.8994, 'JS':-1.2589, 'FID':89.97673588683574, 'IS':2.429979131759416},
{'Model': 'Small', 'Epochs': 125, 'KL': 4.5330, 'Total Variance Distance': 12351.0449, 'JS':-1.5962, 'FID':90.37035669217863, 'IS':2.4083579561346657},
{'Model': 'Small', 'Epochs': 150, 'KL': 6.3408, 'Total Variance Distance': 151500.6562, 'JS':-2.4884, 'FID':93.35474143895999, 'IS':2.3816469392706368},
{'Model': 'Small', 'Epochs': 175, 'KL': 9.0248, 'Total Variance Distance': 1108091.2500, 'JS':-3.8248, 'FID':94.78436847726852, 'IS':2.4010978757710775},
{'Model': 'Small', 'Epochs': 200, 'KL': 11.2292, 'Total Variance Distance': 2993005., 'JS':-4.9347, 'FID':95.67801522152837, 'IS':2.3622274043457097},
{'Model': 'Small', 'Epochs': 225, 'KL': 13.0310, 'Total Variance Distance': 5838262.5, 'JS':-5.8396, 'FID':97.69469469425235, 'IS':2.3862274570377493},
{'Model': 'Small', 'Epochs': 250, 'KL': 14.3435, 'Total Variance Distance': 9227429.0, 'JS':-6.4970, 'FID':99.63915625798319, 'IS':2.363172870000154},

{'Model': 'Medium', 'Epochs': 0,   'KL': -2.0726003503112045,'Total Variance Distance':  0.4213412127953142, 'JS':-0.4846600401891214, 'FID':376.6236941265725, 'IS':1.2348173478412139},
{'Model': 'Medium', 'Epochs': 25,  'KL': 4.3403, 'Total Variance Distance': 82.4296, 'JS':-1.4990, 'FID':118.21016364587604, 'IS':2.0008240688736696},
{'Model': 'Medium', 'Epochs': 50,  'KL': 3.6611, 'Total Variance Distance': 26.6817, 'JS':-1.1727, 'FID':93.13635648600643, 'IS':2.284198851706024},
{'Model': 'Medium', 'Epochs': 75,  'KL': 3.5140, 'Total Variance Distance': 22.9238, 'JS':-1.1042, 'FID':82.61445581224956, 'IS':2.44977795384239},
{'Model': 'Medium', 'Epochs': 100, 'KL': 3.6033, 'Total Variance Distance': 31.7583, 'JS':-1.1477, 'FID':78.95948608339882, 'IS':2.5196854840362928},
{'Model': 'Medium', 'Epochs': 125, 'KL': 4.1846, 'Total Variance Distance': 1314.2395, 'JS':-1.4291, 'FID':80.85456892671675, 'IS':2.507172374461161},
{'Model': 'Medium', 'Epochs': 150, 'KL': 5.5412, 'Total Variance Distance': 37655.5195, 'JS':-2.0960, 'FID':82.50277966485515, 'IS':2.5233912142121553},
{'Model': 'Medium', 'Epochs': 175, 'KL': 7.2260, 'Total Variance Distance': 191063.0938, 'JS':-2.9330, 'FID':85.04552533934293, 'IS':2.4557898208970497},
{'Model': 'Medium', 'Epochs': 200, 'KL': 8.8977, 'Total Variance Distance': 574308.0625, 'JS':-3.7698, 'FID':87.789639391175, 'IS':2.4615996103643694},
{'Model': 'Medium', 'Epochs': 225, 'KL': 9.7625, 'Total Variance Distance': 725594.3750, 'JS':-4.2190, 'FID':90.57263945118541, 'IS':2.4448607219946124},
{'Model': 'Medium', 'Epochs': 250, 'KL': 10.1197, 'Total Variance Distance': 976554.5625, 'JS':-4.4720, 'FID':95.65387186837643, 'IS':2.391574085475964},

{'Model': 'Large', 'Epochs': 0, 'KL': -2.0726003503112045, 'Total Variance Distance': 0.4213412127953142, 'JS':-0.4846600401891214, 'FID':376.6750705733627, 'IS':1.236842138405124},
{'Model': 'Large', 'Epochs': 25, 'KL': 4.0165, 'Total Variance Distance': 42.3675, 'JS':-1.3419, 'FID':118.5399769679521, 'IS':1.9984233112416594},
{'Model': 'Large', 'Epochs': 50, 'KL': 3.5926, 'Total Variance Distance': 24.2609, 'JS':-1.1402, 'FID':89.86348283131366, 'IS':2.3211677423221966},
{'Model': 'Large', 'Epochs': 75, 'KL': 3.4851, 'Total Variance Distance': 21.7810, 'JS':-1.0900, 'FID':79.57145812239372, 'IS':2.507487040344116},
{'Model': 'Large', 'Epochs': 100, 'KL': 3.7381, 'Total Variance Distance': 52.5953, 'JS':-1.2112, 'FID':78.6458192134985, 'IS':2.501101304361945},
{'Model': 'Large', 'Epochs': 125, 'KL': 5.0316, 'Total Variance Distance': 28264.7266, 'JS':-1.8431, 'FID':81.01922441951626, 'IS':2.476919766238947},
{'Model': 'Large', 'Epochs': 150, 'KL': 7.2651, 'Total Variance Distance': 456455.5625, 'JS':-2.9489, 'FID':82.93932793127476, 'IS':2.4598211809621744},
{'Model': 'Large', 'Epochs': 175, 'KL': 9.3410, 'Total Variance Distance': 1072728.5000, 'JS':-3.9840, 'FID':87.29892539976942, 'IS':2.4458626028917907},
{'Model': 'Large', 'Epochs': 200, 'KL': 10.9431, 'Total Variance Distance': 1977501.5000, 'JS':-4.7936, 'FID':94.2688239116095, 'IS':2.369513348128894},
{'Model': 'Large', 'Epochs': 225, 'KL': 10.5780, 'Total Variance Distance': 1503404.6250, 'JS':-4.6937, 'FID':102.37616265868098, 'IS':2.3054664581408617},
{'Model': 'Large', 'Epochs': 250, 'KL': 10.9434, 'Total Variance Distance': 2085890.6250, 'JS':-4.9359, 'FID':111.54278016725766, 'IS':2.2671213042415364},
]

# csv_file = "OnTrainDataResults.csv"
csv_file = "OnTestDataResults.csv"
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in test_rows:
            writer.writerow(data)
except IOError:
    print("I/O error")