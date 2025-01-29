import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os


class DrugDataset:
    """
    A class to load and preprocess the Drug Dataset.
    https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
    """

    def __init__(self):
        self.column_names = [
            'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
            'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh',
            'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semeron', 'VSA'
        ]

    def load_data(self, HOMEFOLDER='/'):
        """Load and preprocess the Drug Dataset."""
        drug = pd.read_csv(os.path.join(HOMEFOLDER,'datasets/drug_consumption.data'), header=None)
        drug.columns = self.column_names
        drug['A'] = (drug['Ethnicity'] == -0.31685) * 1.0
        datasets = []
        As = []
        Ys = []
        for c in drug.Country.unique():
            localdrug = drug[drug.Country == c]
            X = localdrug[self.column_names[1:13]]
            Y = (localdrug['Heroin'] == 'CL0') * 1.0
            A = localdrug.A
            datasets.append((X, Y, A))
            As.append(A.mean())
            Ys.append(Y.mean())
        return datasets

class CompasDataset:
    """
    A class to load and preprocess the Compas Dataset.
    https://www.kaggle.com/danofer/compass
    """
    def __init__(self):
        pass

    def load_data(self, HOMEFOLDER='/'):
        # features to use
        continuous_features = ['age', 'priors_count']
        categorical_features = ['race', 'c_charge_degree', 'sex']
        label = 'two_year_recid'
        sensitive_attribute = 'race'
        client_attribute = 'age_cat'
        # load data
        df = pd.read_csv(os.path.join(HOMEFOLDER,'datasets/compas-scores-two-years.csv'))
        # data filtering
        df = df.dropna(subset=["days_b_screening_arrest"])
        df = df[(df['days_b_screening_arrest']<=30) & (df['days_b_screening_arrest']>=-30)]
        df = df[df['is_recid']!=-1]
        df = df[df['c_charge_degree']!='O']
        df = df[df['score_text']!='NA']
        df = df[(df['race']=='African-American') | (df['race']=='Caucasian')]
        df = df.reset_index()
        # one-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore').fit(df[categorical_features])
        # dataset generation
        datasets = []
        for client in df[client_attribute].unique():
            client_df = df[df[client_attribute]==client]
            X = pd.DataFrame(np.hstack((encoder.transform(client_df[categorical_features]).todense(), client_df[continuous_features])))
            Y = client_df[label]
            A = (client_df[sensitive_attribute]=='African-American')*1.0
            datasets.append((X,Y,A))
        return datasets

class LoanDataset:
    """
    A class to load and preprocess the Loan Dataset.
    https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv
    """

    def __init__(self):
        pass

    def load_data(self):
        """Load and preprocess the Loan Dataset."""
        df = pd.read_csv('datasets/loan.csv')
        columns = [
            'addr_state', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
            'loan_status', 'purpose', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'application_type', 'pub_rec_bankruptcies'
        ]
        df = df[columns]
        df.emp_length.fillna('0 years', inplace=True)
        df.dropna(inplace=True)
        df.term = 1.0 * (df.term == ' 36 months')
        df.sub_grade = df.sub_grade.str[1].astype(float)
        df.emp_length = df.emp_length.str.split(' ').str[-2].str.strip('+').astype(float)
        df.home_ownership = 1.0 * ((df.home_ownership == 'MORTGAGE') + (df.home_ownership == 'OWN'))
        df.issue_d = df.issue_d.str.split('-').str[-1].astype(float)
        df.earliest_cr_line = df.earliest_cr_line.str.split('-').str[-1].astype(float)
        df.initial_list_status = 1.0 * (df.initial_list_status == 'w')
        df.application_type = 1.0 * (df.application_type == 'Individual')
        df['default'] = 1.0 * (df.loan_status == 'Charged Off')
        df = df.loc[(df.loan_status == 'Charged Off') + (df.loan_status == 'Fully Paid')]
        df = df.drop('loan_status', axis=1)
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        df.grade = df.grade.map(grade_mapping)
        Xcolumns_cat = ['verification_status', 'purpose']
        Xcolumns_real = [
            'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_length', 'home_ownership', 'annual_inc',
            'issue_d', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
            'application_type', 'pub_rec_bankruptcies']
        print(df.default.mean())

        datasets = []
        As = []
        Ys = []
        encoder = OneHotEncoder(handle_unknown='ignore').fit(df[Xcolumns_cat])
        for c in df.addr_state.unique():
            localloan = df[df.addr_state == c]
            X = pd.DataFrame(np.hstack((encoder.transform(localloan[Xcolumns_cat]).todense(), localloan[Xcolumns_real])))
            Y = localloan.default
            A = localloan.application_type
            datasets.append((X, Y, A))
            As.append(A.mean())
            Ys.append(Y.mean())
        return datasets


class CommunitiesCrimeDataset:
    """
    A class to load and preprocess the Communities and Crime Dataset.
    https://archive.ics.uci.edu/ml/datasets/communities+and+crime
    """

    def __init__(self):
        pass

    def load_data(self, HOMEFOLDER='/'):
        """Load and preprocess the Communities and Crime Dataset."""
        yvar = 'ViolentCrimesPerPop'
        avar = 'racepctblack'
        with open(os.path.join(HOMEFOLDER,'datasets/communities.names')) as file:
            info = file.read()

        colnames = [line.split(' ')[1] for line in info.split('\n') if line and line.startswith('@attribute')]

        cc = pd.read_csv(os.path.join(HOMEFOLDER,'datasets/communities.data'),
                         header=None,
                         names=colnames,
                         na_values='?')

        nasum = cc.isna().sum()
        names = [name for name in nasum[nasum == 0].index if name not in [yvar, 'state', 'communityname', 'fold']]
        datasets = []
        As = []
        Ys = []
        medianA = cc[avar].median()
        bin_thr = cc[yvar].mean()
        for state in cc.state.unique():
            localcc = cc[cc.state == state]
            X = localcc[names]
            Y = (localcc[yvar] >= bin_thr).astype(int)
            A = (localcc[avar] > medianA).astype(int)
            if len(Y) > 4:
                datasets.append((X, Y, A))
                As.append(A.mean())
                Ys.append(Y.mean())
        return datasets


class SyntheticDataset:
    def __init__(self, K, npoints, nfeatures):
        """
        Constructor for the SyntheticDataset class.

        Args:
        K (int): The number of clients.
        npoints (int or list of ints): Number of datapoints for each client.
                                        If a single integer is provided, all clients will have the same number of datapoints.
                                        If a list of integers is provided, npoints[i] corresponds to the number of datapoints for client i.
        nfeatures (int): The number of features for the dataset.
        additive (bool): Whether the bias is additive (offset) or multiplicative (slope)
        """
        self.K = K
        self.npoints = npoints if isinstance(npoints, list) else [npoints] * K
        self.nfeatures = nfeatures

    def generate_data(self):
        """
        Generate synthetic data for the dataset based on the given parameters.

        Returns:
        datasets (list of tuples): A list of (X, Y, A) pairs for each client, where X, Y, and A are the stacked datapoints
                                    for each client with shapes (npoints, nfeatures), (npoints), and (npoints), respectively.
        """
        center = np.ones(self.nfeatures)
        m = np.ones(self.nfeatures)

        datasets = []
        for i in range(self.K):
            X0 = np.random.multivariate_normal((2*np.mod(i,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X1 = np.random.multivariate_normal((2*np.mod(i+1,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X = np.vstack((X0,X1))
            A = np.concatenate((np.ones(int(self.npoints[i]/2)), np.zeros(int(self.npoints[i]/2))))
            Y = (np.dot(X, m)>= 0).astype(int)
            datasets.append((pd.DataFrame(np.hstack((X,np.expand_dims(A,1)))), pd.Series(Y), pd.Series(A)))
        return datasets

class HeterogenityDataset:
    def __init__(self, K, npoints, nfeatures, heterogenity):
        """
        Constructor for the SyntheticDataset class.

        Args:
        K (int): The number of clients.
        npoints (int or list of ints): Number of datapoints for each client.
                                        If a single integer is provided, all clients will have the same number of datapoints.
                                        If a list of integers is provided, npoints[i] corresponds to the number of datapoints for client i.
        nfeatures (int): The number of features for the dataset.
        heterogenity (float): Mixture weight for heterogenity: 0.5 (full homogeneous) to 1.0 (full heterogenious)

        """
        self.K = K
        self.npoints = npoints if isinstance(npoints, list) else [npoints] * K
        self.nfeatures = nfeatures
        self.heterogenity = heterogenity

    def generate_data(self):
        """
        Generate synthetic data for the dataset based on the given parameters.

        Returns:
        datasets (list of tuples): A list of (X, Y, A) pairs for each client, where X, Y, and A are the stacked datapoints
                                    for each client with shapes (npoints, nfeatures), (npoints), and (npoints), respectively.
        """
        center = np.ones(self.nfeatures)
        m = np.ones(self.nfeatures)

        datasets = []
        for i in range(self.K):
            X0_alpha1 = np.random.multivariate_normal((2*np.mod(i,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X0_alpha0 = np.random.multivariate_normal((2*np.mod(i+1,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            alpha_X0 = np.random.binomial(1, self.heterogenity, (int(self.npoints[i]/2),1))
            X0 = alpha_X0 * X0_alpha1 + (1 - alpha_X0) * X0_alpha0

            X1_alpha1 = np.random.multivariate_normal((2*np.mod(i+1,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X1_alpha0 = np.random.multivariate_normal((2*np.mod(i,2)-1)*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            alpha_X1 = np.random.binomial(1, self.heterogenity, (int(self.npoints[i]/2),1))
            X1 = alpha_X1 * X1_alpha1 + (1 - alpha_X1) * X1_alpha0

            X = np.vstack((X0,X1))
            A = np.concatenate((np.ones(int(self.npoints[i]/2)), np.zeros(int(self.npoints[i]/2))))
            Y = (np.dot(X, m)>= 0).astype(int)
            datasets.append((pd.DataFrame(np.hstack((X,np.expand_dims(A,1)))), pd.Series(Y), pd.Series(A)))
        return datasets


class HeterogenityDatasetv2:
    def __init__(self, heterogenity):
        """
        Constructor for the SyntheticDataset class.

        Args:
        heterogenity (float): Mixture weight for heterogenity: 0.5 (full homogeneous) to 1.0 (full heterogenious)

        """
        self.npoints = [500, 500, 200, 200, 200, 200, 200]
        self.nfeatures = 10
        self.K = 7
        self.heterogenity = heterogenity

    def generate_data(self):
        """
        Generate synthetic data for the dataset based on the given parameters.

        Returns:
        datasets (list of tuples): A list of (X, Y, A) pairs for each client, where X, Y, and A are the stacked datapoints
                                    for each client with shapes (npoints, nfeatures), (npoints), and (npoints), respectively.
        """
        center = np.ones(self.nfeatures)
        m = np.ones(self.nfeatures)

        datasets = []
        for i in range(6):
            c0 = -1 if i<=1 else 1
            c1 = -1*c0

            X0_alpha1 = np.random.multivariate_normal(c0*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X0_alpha0 = np.random.multivariate_normal(c1*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            alpha_X0 = np.random.binomial(1, self.heterogenity, (int(self.npoints[i]/2),1))
            X0 = alpha_X0 * X0_alpha1 + (1 - alpha_X0) * X0_alpha0

            X1_alpha1 = np.random.multivariate_normal(c1*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            X1_alpha0 = np.random.multivariate_normal(c0*center, np.identity(self.nfeatures), int(self.npoints[i]/2))
            alpha_X1 = np.random.binomial(1, self.heterogenity, (int(self.npoints[i]/2),1))
            X1 = alpha_X1 * X1_alpha1 + (1 - alpha_X1) * X1_alpha0

            X = np.vstack((X0,X1))
            A = np.concatenate((np.ones(int(self.npoints[i]/2)), np.zeros(int(self.npoints[i]/2))))
            Y = (np.dot(X, m)>= 0).astype(int)
            datasets.append((pd.DataFrame(np.hstack((X,np.expand_dims(A,1)))), pd.Series(Y), pd.Series(A)))
            #datasets.append((pd.DataFrame(X), pd.Series(Y), pd.Series(A)))
        return datasets

if __name__ == "__main__":
    drug_dataset = DrugDataset()
    drug_data = drug_dataset.load_data()
    print('Drug Dataset')
    for X, Y, A in drug_data:
        print(f'A: {A.mean():.2f}\t Y: {Y.mean():.2f} N: {len(A)}')

    print('-' * 5)
    print('-' * 5)
    loan_dataset = LoanDataset()
    loan_data = loan_dataset.load_data()
    print('Loan Dataset')
    for X, Y, A in loan_data:
        print(f'A: {A.mean():.2f}\t Y: {Y.mean():.2f} N: {len(A)}')

    print('-' * 5)
    print('-' * 5)
    communities_crime_dataset = CommunitiesCrimeDataset()
    communities_crime_data = communities_crime_dataset.load_data()
    print('Communities/Crime Dataset')
    for X, Y, A in communities_crime_data:
        print(f'A: {A.mean():.2f}\t Y: {Y.mean():.2f} N: {len(A)}')
