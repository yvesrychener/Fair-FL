import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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

    def load_data(self):
        """Load and preprocess the Drug Dataset."""
        drug = pd.read_csv('datasets/drug_consumption.data', header=None)
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

    def load_data(self):
        """Load and preprocess the Communities and Crime Dataset."""
        yvar = 'ViolentCrimesPerPop'
        avar = 'racepctblack'
        with open('datasets/communities.names') as file:
            info = file.read()

        colnames = [line.split(' ')[1] for line in info.split('\n') if line and line.startswith('@attribute')]

        cc = pd.read_csv('datasets/communities.data',
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
    def __init__(self, kappa, npoints, nfeatures):
        """
        Constructor for the SyntheticDataset class.

        Args:
        kappa (int): The number of clients.
        npoints (int or list of ints): Number of datapoints for each client.
                                        If a single integer is provided, all clients will have the same number of datapoints.
                                        If a list of integers is provided, npoints[i] corresponds to the number of datapoints for client i.
        nfeatures (int): The number of features for the dataset.
        """
        self.kappa = kappa
        self.npoints = npoints if isinstance(npoints, list) else [npoints] * kappa
        self.nfeatures = nfeatures

    def generate_data(self):
        """
        Generate synthetic data for the dataset based on the given parameters.

        Returns:
        datasets (list of tuples): A list of (X, Y, A) pairs for each client, where X, Y, and A are the stacked datapoints
                                    for each client with shapes (npoints, nfeatures), (npoints), and (npoints), respectively.
        """
        centers = np.random.multivariate_normal(np.zeros(self.nfeatures), np.identity(self.nfeatures), self.kappa)
        m = np.random.randn(self.nfeatures)

        datasets = []
        for i in range(self.kappa):
            center = centers[i]
            X = np.random.multivariate_normal(center, np.identity(self.nfeatures), self.npoints[i])
            A = np.random.randint(0, 2, self.npoints[i])
            Y = (np.dot(X, m) >= 0).astype(int)
            datasets.append((X, Y, A))

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
