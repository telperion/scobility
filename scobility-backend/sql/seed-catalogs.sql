MERGE dbo.Catalog AS t
USING (VALUES
    (1, N'StepManiaX', 1, CAST(N'2024-02-29T00:00:00' AS DATETIME2), 0.003, 100000.0),
    (2, N'ITL2022', 0, CAST(N'2024-02-29T00:00:00' AS DATETIME2), 0.003, 10000.0),
    (3, N'ITL2023', 1, CAST(N'2024-02-29T00:00:00' AS DATETIME2), 0.003, 10000.0),
    (4, N'ITL2024', 1, CAST(N'2024-02-29T00:00:00' AS DATETIME2), 0.003, 10000.0),
    (5, N'ITL2025', 1, CAST(N'2025-02-28T00:00:00' AS DATETIME2), 0.003, 10000.0),
    (6, N'ITL2026', 1, CAST(N'2026-02-28T00:00:00' AS DATETIME2), 0.003, 10000.0)
) AS s (catalog_id, name, active, last_update, perfect_offset, perfect_score)
ON t.catalog_id = s.catalog_id
WHEN MATCHED THEN UPDATE SET
    name = s.name,
    active = s.active,
    last_update = s.last_update,
    perfect_offset = s.perfect_offset,
    perfect_score = s.perfect_score
WHEN NOT MATCHED THEN INSERT (catalog_id, name, active, last_update, perfect_offset, perfect_score)
    VALUES (s.catalog_id, s.name, s.active, s.last_update, s.perfect_offset, s.perfect_score);
GO