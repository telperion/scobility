IF OBJECT_ID(N'dbo.Score', N'U') IS NOT NULL DROP TABLE dbo.Score;
IF OBJECT_ID(N'dbo.Relationship', N'U') IS NOT NULL DROP TABLE dbo.Relationship;
IF OBJECT_ID(N'dbo.Player', N'U') IS NOT NULL DROP TABLE dbo.Player;
IF OBJECT_ID(N'dbo.Chart', N'U') IS NOT NULL DROP TABLE dbo.Chart;
IF OBJECT_ID(N'dbo.Catalog', N'U') IS NOT NULL DROP TABLE dbo.Catalog;
GO

CREATE TABLE dbo.Catalog (
    catalog_id INT NOT NULL PRIMARY KEY,
    name NVARCHAR(128) NOT NULL,
    active BIT NOT NULL DEFAULT 1,
    last_update DATETIME2 NULL,
    perfect_offset FLOAT NULL,
    perfect_score FLOAT NULL
);

CREATE TABLE dbo.Chart (
    global_chart_id INT NOT NULL PRIMARY KEY,
    catalog_id INT NOT NULL,
    chart_id INT NOT NULL,
    hash NVARCHAR(32) NOT NULL,
    title NVARCHAR(512) NULL,
    subtitle NVARCHAR(512) NULL,
    artist NVARCHAR(512) NULL,
    meter FLOAT NULL,
    slot NVARCHAR(64) NULL,
    style NVARCHAR(32) NULL,
    value FLOAT NULL,
    value_scoring FLOAT NULL,
    value_passing FLOAT NULL,
    spice FLOAT NULL,
    spice_calc_time DATETIME2 NULL,
    CONSTRAINT FK_Chart_Catalog FOREIGN KEY (catalog_id) REFERENCES dbo.Catalog(catalog_id)
);

CREATE INDEX IX_Chart_catalog ON dbo.Chart(catalog_id);
CREATE INDEX IX_Chart_catalog_chart ON dbo.Chart(catalog_id, chart_id);

CREATE TABLE dbo.Player (
    performance_id INT NOT NULL PRIMARY KEY,
    catalog_id INT NOT NULL,
    entrant_id INT NOT NULL,
    groovestats_id INT NULL,
    boogiestats_id INT NULL,
    name NVARCHAR(256) NULL,
    scobility FLOAT NULL,
    timing_power FLOAT NULL,
    comfort_zone FLOAT NULL,
    scobility_calc_time DATETIME2 NULL,
    CONSTRAINT FK_Player_Catalog FOREIGN KEY (catalog_id) REFERENCES dbo.Catalog(catalog_id)
);

CREATE INDEX IX_Player_catalog ON dbo.Player(catalog_id);
CREATE INDEX IX_Player_entrant ON dbo.Player(catalog_id, entrant_id);

CREATE TABLE dbo.Score (
    score_id BIGINT NOT NULL PRIMARY KEY,
    catalog_id INT NOT NULL,
    global_chart_id INT NOT NULL,
    performance_id INT NOT NULL,
    plays INT NULL,
    last_played DATETIME2 NULL,
    clear NVARCHAR(32) NULL,
    score FLOAT NULL,
    prediction FLOAT NULL,
    prediction_calc_time DATETIME2 NULL,
    CONSTRAINT FK_Score_Catalog FOREIGN KEY (catalog_id) REFERENCES dbo.Catalog(catalog_id),
    CONSTRAINT FK_Score_Chart FOREIGN KEY (global_chart_id) REFERENCES dbo.Chart(global_chart_id),
    CONSTRAINT FK_Score_Player FOREIGN KEY (performance_id) REFERENCES dbo.Player(performance_id)
);

CREATE INDEX IX_Score_catalog ON dbo.Score(catalog_id);
CREATE INDEX IX_Score_performance ON dbo.Score(catalog_id, performance_id);

CREATE TABLE dbo.Relationship (
    relationship_id BIGINT NOT NULL PRIMARY KEY,
    x_id INT NOT NULL,
    y_id INT NOT NULL,
    common INT NULL,
    relation FLOAT NULL,
    strength FLOAT NULL
);
GO