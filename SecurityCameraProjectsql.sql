-- Code ile uyumlu tablo ve db kurulumunu yaptýk 
CREATE TABLE VehicleDetections (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    VehicleType VARCHAR(50) NOT NULL,
    Confidence FLOAT NOT NULL,
    BoundingBox VARCHAR(100) NOT NULL,
    DetectedAt DATETIME DEFAULT GETDATE()
);

-- veri tabanina kayýt ettiðimiz verilerimizi çekip kullanýcýya gösterdik
SELECT TOP 50 * FROM VehicleDetections ORDER BY Id DESC;