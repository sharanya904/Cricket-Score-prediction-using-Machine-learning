


#importing libraries 
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import preprocessing 


class MyModel:
    def __init__(self):
        self._model = GradientBoostingRegressor(learning_rate=0.5)
        
    
   
    
    def fit(self, files):
        #cleaning
        df_res=files[1]  #match_results_file
        df=files[0]   #ball by ball
        
        df_res = df_res.iloc[::-1]
        
        df5 = df.loc[df['overs']<6]
        df2 =  df5.loc[df['innings']<3]
        
        #electing data according to the requirements mentioned(columns)
        df_resu = df_res[['ID','Team1', 'Team2','Venue']]
        df_cat = df2[['ID','innings', 'batter', 'bowler', 'BattingTeam']]
        df_num = df2[['ID','innings', 'total_run']]
        df1 = df_cat.groupby(["ID", "innings",]).agg(set)
        df3 = df_num.groupby(["ID", "innings"]).sum()
        
        #converting data to list 
        ade = df_resu[['Venue']].values.tolist()
        idd = df_resu[['ID']].values.tolist()

        v = []
        d = []
        for i in range(len(ade)):
            v.append(ade[i])
            d.append(idd[i])

        df_ven = pd.DataFrame(d, columns = ['ID'])
        df_ven['Venue'] = v

        abc = df1[['batter']].values.tolist()
        o = []
        for i in range(len(abc)):
            o.append(len(abc[i][0])-2)
        df1['out'] = o  
        bt = df1[['BattingTeam']].values.tolist()

        for i in range(0,len(bt),2):
            temp = bt[i+1]
            bt[i+1] = bt[i]
            bt[i] = temp
        df1['BowlingTeam'] = bt

        res = pd.concat([df1, df3], axis=1) 

        self.final = pd.merge(res, df_ven, on=['ID'], how = 'left')  ######finalllllll
        idd = self.final[['ID']].values.tolist()

        BatTeam = self.final['BattingTeam'].values.tolist()
        BT = []
        for i in BatTeam:
            BT.append(list(i)[0])
        self.final['BattingTeam'] = BT
    
        BowlTeam = self.final['BowlingTeam'].values
        BW = []
        for i in BowlTeam:
            BW.append(list(i[0])[0])
        self.final['BowlingTeam'] = BW


        Venu = self.final['Venue'].values.tolist()
        Ve = []
        for i in Venu:
            Ve.append(i[0])
        self.final['Venue'] = Ve

        Bat = self.final['batter'].values.tolist()
        BAT = []
        for i in Bat:
            BAT.append(list(i))
        self.final['batter'] = BAT

        Bowl = self.final['bowler'].values.tolist()
        BOWL = []
        for i in Bowl:
            BOWL.append(list(i))
        self.final['bowler'] = BOWL
        
        df = self.final.copy()
        df1 = self.final.copy()
        
        df.drop(df.columns[[1,2]],axis=1,inplace = True)
        df1.drop(df1.columns[[1,2]],axis=1,inplace = True)
        
        df.loc[df["BattingTeam"] == "Rising Pune Supergiant", "BattingTeam"] = 'Rising Pune Supergiants'
        df.loc[df["BattingTeam"] == "Delhi Daredevils", "BattingTeam"] = 'Delhi Capitals'
        df.loc[df["BattingTeam"] == "Gujarat Lions", "BattingTeam"] = 'Gujarat Titans'
        df.loc[df["BattingTeam"] == "Kings XI Punjab", "BattingTeam"] = 'Punjab Kings'

        df1.loc[df1["BattingTeam"] == "Rising Pune Supergiant", "BattingTeam"] = 'Rising Pune Supergiants'
        df1.loc[df1["BattingTeam"] == "Delhi Daredevils", "BattingTeam"] = 'Delhi Capitals'
        df1.loc[df1["BattingTeam"] == "Gujarat Lions", "BattingTeam"] = 'Gujarat Titans'
        df1.loc[df1["BattingTeam"] == "Kings XI Punjab", "BattingTeam"] = 'Punjab Kings'

        label_encoder = preprocessing.LabelEncoder()

        df1['BattingTeam'] = label_encoder.fit_transform(df1['BattingTeam'])
        df1['BowlingTeam'] = label_encoder.fit_transform(df1['BowlingTeam'])
        df1['Venue'] = label_encoder.fit_transform(df1['Venue'])
        
        X = df1[['BattingTeam', 'out', 'BowlingTeam', 'Venue']]
        y = df1['total_run']

        regr = linear_model.LinearRegression()
        regr.fit(X, y)

        predicted_runs = regr.predict(df1[['BattingTeam', 'out', 'BowlingTeam', 'Venue']])
        df['predicted_runs'] = predicted_runs
        
        final_data = pd.get_dummies(df)
        
        labels = np.array(final_data['total_run'])

        # Remove the labels from the features
        final_data = final_data.drop('total_run', axis = 1)
        final_data.drop(final_data.columns[[0,2]],axis=1,inplace = True)

        # Create empty DataFrame with those column names
        names = [x for x in final_data.columns]
        df_b = pd.DataFrame(columns=names)
        
        df_b.loc[len(df_b.index)] = [0 for x in range(82)]
        df_b.loc[len(df_b.index)] = [0 for x in range(82)]
        self.emp_input = df_b
    
        # Convert to numpy array
        final_data = np.array(final_data)
        
        self._model.fit(final_data, labels)
        
    def predict(self, test_data):
        self.pred_data = test_data
        abc = self.pred_data[['batsmen']].values.tolist()
        o = []
        for i in range(len(abc)):
            u = len(abc[i][0].split(sep = ','))-2
            o.append(u)
        self.pred_data['out'] = o
        self.emp_input['out'] = o
        
        #changing the input format to desired format
        self.emp_input[f'BattingTeam_{self.pred_data["batting_team"][0]}'][0] = 1
        self.emp_input[f'BowlingTeam_{self.pred_data["bowling_team"][0]}'][0] = 1

        self.emp_input[f'BattingTeam_{self.pred_data["batting_team"][1]}'][1] = 1
        self.emp_input[f'BowlingTeam_{self.pred_data["bowling_team"][1]}'][1] = 1
        #self.emp_input.drop(self.emp_input.columns[[0,2]],axis=1,inplace = True)
        
        self.emp_input = np.array(self.emp_input)  #######

        predict = self._model.predict(self.emp_input)
        
        pred = []
        pred.append(int(predict[0]))
        pred.append(int(predict[1]))
#         self.submission['predicted_runs'][0] = int(predict[0])
#         self.submission['predicted_runs'][1] = int(predict[1])
        #self.submission.to_csv('submission.csv')
        
        return pred

