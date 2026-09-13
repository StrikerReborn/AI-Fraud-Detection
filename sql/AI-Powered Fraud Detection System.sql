USE FraudDetectionDB;
GO

CREATE TABLE FraudTransactions
(
    TransactionID INT IDENTITY(1,1) PRIMARY KEY,

    Step INT,

    TransactionType VARCHAR(20),

    Amount FLOAT,

    OldBalanceOrg FLOAT,

    OldBalanceDest FLOAT,

    IsFlaggedFraud INT,

    Hour INT,

    PreviousAmount FLOAT,

    TimeSincePrevious FLOAT,

    PriorTransactionCount INT,

    PriorAverageAmount FLOAT,

    AmountVsPreviousAverage FLOAT,

    HasPreviousTransaction INT,

    FraudProbability FLOAT,

    RiskScore FLOAT,

    RiskLevel VARCHAR(20),

    Decision VARCHAR(30),

    PredictionTime DATETIME DEFAULT GETDATE()
);
GO

SELECT *
FROM FraudTransactions;


SELECT TOP 1 *
FROM FraudTransactions
ORDER BY TransactionID DESC;