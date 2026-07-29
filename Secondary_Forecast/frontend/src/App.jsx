import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertCircle,
  Boxes,
  CalendarRange,
  ChevronDown,
  ChevronUp,
  Filter,
  Package,
  RefreshCw,
  Search,
  Truck,
  TrendingUp,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';
const HORIZONS = [30, 60, 90];
const PAGE_SIZE = 12;
const EMPTY_SKU_METADATA = {
  detectedColumns: {},
  itemsById: {},
};
const EMPTY_DISTRIBUTOR_METADATA = {
  detectedColumns: {},
  itemsById: {},
};

const formatQuantity = (value) => {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return '0';
  }

  return new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 0,
  }).format(numericValue);
};

const formatMonth = (dateString) => {
  if (!dateString) {
    return 'Not available';
  }

  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return dateString;
  }

  return date.toLocaleDateString('en-US', {
    month: 'short',
    year: 'numeric',
  });
};

const formatDateTime = (dateString) => {
  if (!dateString) {
    return 'Unavailable';
  }

  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return 'Unavailable';
  }

  return date.toLocaleString('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
};

const toNumber = (value) => {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : 0;
};

const sortStrings = (values) =>
  [...values].sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
  );

const getSortableValue = (row, key) => {
  if (key === 'predicted_revenue' || key === 'P10' || key === 'P90' || key === 'horizon_days') {
    return toNumber(row[key]);
  }

  return String(row[key] ?? '');
};

const getConfidenceMetadata = (records) => {
  const sample = records.find((record) => record);
  if (!sample) {
    return {
      subtext: 'No forecast rows available for confidence interval analysis',
    };
  }

  const hasDistinctBounds = records.some((record) => {
    const predicted = toNumber(record.predicted_revenue);
    return toNumber(record.P10) !== predicted || toNumber(record.P90) !== predicted;
  });

  return {
    subtext: hasDistinctBounds
      ? 'P10 and P90 reflect the lower and upper confidence bounds returned by the API'
      : 'The current backend is returning identical values for prediction, P10, and P90',
  };
};

const getSkuInfo = (skuMetadata, skuId) => {
  return skuMetadata.itemsById?.[String(skuId)] ?? null;
};

const getSkuDisplayLabel = (skuMetadata, skuId) => {
  const skuInfo = getSkuInfo(skuMetadata, skuId);
  if (!skuInfo) {
    return String(skuId ?? 'Unknown SKU');
  }

  return skuInfo.bestIdentifier || skuInfo.displayName || skuInfo.productName || skuInfo.skuCode || skuInfo.skuId;
};

const getDistributorInfo = (distributorMetadata, distributorId) => {
  return distributorMetadata.itemsById?.[String(distributorId)] ?? null;
};

const getDistributorDisplayLabel = (distributorMetadata, distributorId) => {
  const distributorInfo = getDistributorInfo(distributorMetadata, distributorId);
  if (!distributorInfo) {
    return String(distributorId ?? 'Unknown Distributor');
  }

  return (
    distributorInfo.bestIdentifier ||
    distributorInfo.displayName ||
    distributorInfo.distributorName ||
    distributorInfo.distributorCode ||
    distributorInfo.distributorId
  );
};

export default function App() {
  const [records, setRecords] = useState([]);
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [skuMetadata, setSkuMetadata] = useState(EMPTY_SKU_METADATA);
  const [distributorMetadata, setDistributorMetadata] = useState(EMPTY_DISTRIBUTOR_METADATA);
  const [lastTrainingTime, setLastTrainingTime] = useState('');
  const [selectedDistributor, setSelectedDistributor] = useState('ALL');
  const [selectedSku, setSelectedSku] = useState('ALL');
  const [selectedHorizon, setSelectedHorizon] = useState('ALL');
  const [selectedMonth, setSelectedMonth] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortConfig, setSortConfig] = useState({
    key: 'forecast_month',
    direction: 'asc',
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const loadDashboard = async () => {
    setError('');
    setRefreshing(true);

    try {
      const [forecastResponse, statusResponse, metricsResponse, skuMetadataResponse, distributorMetadataResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/forecast`),
        fetch(`${API_BASE_URL}/api/models/forecast/status`),
        fetch(`${API_BASE_URL}/api/models/forecast/metrics`),
        fetch('/sku-metadata.json'),
        fetch('/distributor-metadata.json'),
      ]);

      if (!forecastResponse.ok) {
        throw new Error(await forecastResponse.text() || `Forecast API error ${forecastResponse.status}`);
      }

      if (!statusResponse.ok) {
        throw new Error(await statusResponse.text() || `Status API error ${statusResponse.status}`);
      }

      if (!metricsResponse.ok) {
        throw new Error(await metricsResponse.text() || `Metrics API error ${metricsResponse.status}`);
      }

      if (!skuMetadataResponse.ok) {
        throw new Error(await skuMetadataResponse.text() || `SKU metadata error ${skuMetadataResponse.status}`);
      }

      if (!distributorMetadataResponse.ok) {
        throw new Error(
          await distributorMetadataResponse.text() ||
            `Distributor metadata error ${distributorMetadataResponse.status}`
        );
      }

      const [forecastJson, statusJson, metricsJson, skuMetadataJson, distributorMetadataJson] = await Promise.all([
        forecastResponse.json(),
        statusResponse.json(),
        metricsResponse.json(),
        skuMetadataResponse.json(),
        distributorMetadataResponse.json(),
      ]);

      const metricsLastModified =
        metricsResponse.headers.get('last-modified') ??
        metricsResponse.headers.get('date') ??
        '';

      setRecords(Array.isArray(forecastJson) ? forecastJson : []);
      setStatus(statusJson ?? null);
      setMetrics(metricsJson ?? null);
      setSkuMetadata(skuMetadataJson ?? EMPTY_SKU_METADATA);
      setDistributorMetadata(distributorMetadataJson ?? EMPTY_DISTRIBUTOR_METADATA);
      setLastTrainingTime(metricsLastModified);
    } catch (loadError) {
      console.error(loadError);
      setError(loadError instanceof Error ? loadError.message : 'Unable to load Secondary Sales data');
      setRecords([]);
      setStatus(null);
      setMetrics(null);
      setSkuMetadata(EMPTY_SKU_METADATA);
      setDistributorMetadata(EMPTY_DISTRIBUTOR_METADATA);
      setLastTrainingTime('');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  const distributorOptions = useMemo(
    () =>
      Array.from(new Set(records.map((record) => String(record.payer_code ?? '')).filter(Boolean)))
        .map((distributorId) => ({
          value: distributorId,
          label: getDistributorDisplayLabel(distributorMetadata, distributorId),
        }))
        .sort((left, right) =>
          left.label.localeCompare(right.label, undefined, { numeric: true, sensitivity: 'base' })
        ),
    [distributorMetadata, records]
  );

  const skuOptions = useMemo(() => {
    const uniqueSkuIds = Array.from(new Set(records.map((record) => String(record.pack_type ?? '')).filter(Boolean)));

    return uniqueSkuIds
      .map((skuId) => ({
        value: skuId,
        label: getSkuDisplayLabel(skuMetadata, skuId),
      }))
      .sort((left, right) =>
        left.label.localeCompare(right.label, undefined, { numeric: true, sensitivity: 'base' })
      );
  }, [records, skuMetadata]);

  const monthOptions = useMemo(
    () =>
      sortStrings(
        Array.from(new Set(records.map((record) => String(record.forecast_month ?? '')).filter(Boolean)))
      ),
    [records]
  );

  const filteredRecords = useMemo(
    () =>
      records.filter((record) => {
        const distributorMatch =
          selectedDistributor === 'ALL' || String(record.payer_code) === selectedDistributor;
        const skuMatch = selectedSku === 'ALL' || String(record.pack_type) === selectedSku;
        const horizonMatch =
          selectedHorizon === 'ALL' || Number(record.horizon_days) === Number(selectedHorizon);
        const monthMatch = selectedMonth === 'ALL' || String(record.forecast_month) === selectedMonth;

        return distributorMatch && skuMatch && horizonMatch && monthMatch;
      }),
    [records, selectedDistributor, selectedSku, selectedHorizon, selectedMonth]
  );

  const searchableRows = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    if (!normalizedQuery) {
      return filteredRecords;
    }

    return filteredRecords.filter((row) => {
      const skuInfo = getSkuInfo(skuMetadata, row.pack_type);
      const searchValues = [
        row.payer_code,
        row.pack_type,
        row.forecast_month,
        `${row.horizon_days}`,
        `${row.predicted_revenue}`,
        `${row.P10}`,
        `${row.P90}`,
        getDistributorInfo(distributorMetadata, row.payer_code)?.distributorName,
        getDistributorInfo(distributorMetadata, row.payer_code)?.displayName,
        getDistributorInfo(distributorMetadata, row.payer_code)?.distributorCode,
        getDistributorInfo(distributorMetadata, row.payer_code)?.stateName,
        getDistributorInfo(distributorMetadata, row.payer_code)?.cityName,
        skuInfo?.productName,
        skuInfo?.displayName,
        skuInfo?.bestIdentifier,
        skuInfo?.skuCode,
        skuInfo?.brand,
        skuInfo?.category,
        skuInfo?.subcategory,
      ];

      return searchValues.some((value) => String(value ?? '').toLowerCase().includes(normalizedQuery));
    });
  }, [distributorMetadata, filteredRecords, searchQuery, skuMetadata]);

  const comparisonCards = useMemo(
    () =>
      HORIZONS.map((horizon) => {
        const horizonRecords = filteredRecords.filter(
          (record) => Number(record.horizon_days) === horizon
        );

        const totalQuantity = horizonRecords.reduce(
          (sum, record) => sum + toNumber(record.predicted_revenue),
          0
        );
        const lowerBound = horizonRecords.reduce((sum, record) => sum + toNumber(record.P10), 0);
        const upperBound = horizonRecords.reduce((sum, record) => sum + toNumber(record.P90), 0);
        const uniqueMonths = new Set(
          horizonRecords.map((record) => String(record.forecast_month ?? '')).filter(Boolean)
        ).size;

        return {
          horizon,
          totalQuantity,
          lowerBound,
          upperBound,
          uniqueMonths,
        };
      }),
    [filteredRecords]
  );

  const chartData = useMemo(() => {
    const monthMap = new Map();

    filteredRecords.forEach((record) => {
      const forecastMonth = String(record.forecast_month ?? '');
      if (!forecastMonth) {
        return;
      }

      if (!monthMap.has(forecastMonth)) {
        monthMap.set(forecastMonth, {
          forecastMonth,
          label: formatMonth(forecastMonth),
          qty30: null,
          qty60: null,
          qty90: null,
        });
      }

      const monthEntry = monthMap.get(forecastMonth);
      const horizon = Number(record.horizon_days);
      const predictedQuantity = toNumber(record.predicted_revenue);

      if (horizon === 30) {
        monthEntry.qty30 = (monthEntry.qty30 ?? 0) + predictedQuantity;
      }
      if (horizon === 60) {
        monthEntry.qty60 = (monthEntry.qty60 ?? 0) + predictedQuantity;
      }
      if (horizon === 90) {
        monthEntry.qty90 = (monthEntry.qty90 ?? 0) + predictedQuantity;
      }
    });

    return Array.from(monthMap.values()).sort((left, right) =>
      left.forecastMonth.localeCompare(right.forecastMonth)
    );
  }, [filteredRecords]);

  const tableColumnAvailability = useMemo(() => {
    const brandColumn = skuMetadata.detectedColumns?.brand;
    const categoryColumn = skuMetadata.detectedColumns?.category;

    return {
      showBrand: Boolean(
        brandColumn &&
          filteredRecords.some((record) => {
            const skuInfo = getSkuInfo(skuMetadata, record.pack_type);
            return skuInfo?.brand;
          })
      ),
      showCategory: Boolean(
        categoryColumn &&
          filteredRecords.some((record) => {
            const skuInfo = getSkuInfo(skuMetadata, record.pack_type);
            return skuInfo?.category;
          })
      ),
    };
  }, [filteredRecords, skuMetadata]);

  const summaryCards = useMemo(() => {
    const totalPredictedRevenue = filteredRecords.reduce(
      (sum, record) => sum + toNumber(record.predicted_revenue),
      0
    );
    const distributorTotals = new Map();
    const skuTotals = new Map();

    filteredRecords.forEach((record) => {
      const distributorKey = String(record.payer_code ?? '');
      const skuKey = String(record.pack_type ?? '');
      const amount = toNumber(record.predicted_revenue);

      distributorTotals.set(distributorKey, (distributorTotals.get(distributorKey) ?? 0) + amount);
      skuTotals.set(skuKey, (skuTotals.get(skuKey) ?? 0) + amount);
    });

    const distributorCount = distributorTotals.size;
    const averageQuantityPerDistributor =
      distributorCount > 0 ? totalPredictedRevenue / distributorCount : 0;

    const topDistributorEntry =
      [...distributorTotals.entries()].sort((left, right) => right[1] - left[1])[0] ?? [];
    const topSkuEntry = [...skuTotals.entries()].sort((left, right) => right[1] - left[1])[0] ?? [];
    const metricKeys = metrics ? Object.keys(metrics) : [];
    const confidenceMetadata = getConfidenceMetadata(filteredRecords);

    return [
      {
        title: 'Total Predicted Revenue',
        value: formatQuantity(totalPredictedRevenue),
        subtext: 'Based on the backend `predicted_revenue` field for the current filter set',
        icon: TrendingUp,
      },
      {
        title: 'Average Quantity per Distributor',
        value: formatQuantity(averageQuantityPerDistributor),
        subtext: `${formatQuantity(distributorCount)} distributor${distributorCount === 1 ? '' : 's'} in view`,
        icon: Truck,
      },
      {
        title: 'Top Forecast SKU',
        value: topSkuEntry[0] ? getSkuDisplayLabel(skuMetadata, topSkuEntry[0]) : 'Unavailable',
        subtext: topSkuEntry[1] ? `Predicted quantity: ${formatQuantity(topSkuEntry[1])}` : 'No SKU data in current view',
        icon: Package,
      },
      {
        title: 'Top Forecast Distributor',
        value: topDistributorEntry[0]
          ? getDistributorDisplayLabel(distributorMetadata, topDistributorEntry[0])
          : 'Unavailable',
        subtext: topDistributorEntry[1]
          ? `Predicted quantity: ${formatQuantity(topDistributorEntry[1])}`
          : 'No distributor data in current view',
        icon: Boxes,
      },
      {
        title: 'Last Model Training Time',
        value: formatDateTime(lastTrainingTime),
        subtext: lastTrainingTime
          ? 'Derived from the forecast metrics response metadata'
          : 'Training timestamp is not exposed by the current backend',
        icon: Activity,
      },
      {
        title: 'Forecast Coverage',
        value: formatQuantity(metricKeys.length || new Set(filteredRecords.map((row) => row.forecast_month)).size),
        subtext: confidenceMetadata.subtext,
        icon: CalendarRange,
      },
    ];
  }, [distributorMetadata, filteredRecords, lastTrainingTime, metrics, skuMetadata]);

  const sortedRows = useMemo(() => {
    const rows = [...searchableRows];
    rows.sort((left, right) => {
      const leftValue =
        sortConfig.key === 'distributor_label'
          ? getDistributorDisplayLabel(distributorMetadata, left.payer_code)
          :
        sortConfig.key === 'product_label'
          ? getSkuDisplayLabel(skuMetadata, left.pack_type)
          : sortConfig.key === 'brand'
            ? getSkuInfo(skuMetadata, left.pack_type)?.brand ?? ''
            : sortConfig.key === 'category'
              ? getSkuInfo(skuMetadata, left.pack_type)?.category ?? ''
              : getSortableValue(left, sortConfig.key);

      const rightValue =
        sortConfig.key === 'distributor_label'
          ? getDistributorDisplayLabel(distributorMetadata, right.payer_code)
          :
        sortConfig.key === 'product_label'
          ? getSkuDisplayLabel(skuMetadata, right.pack_type)
          : sortConfig.key === 'brand'
            ? getSkuInfo(skuMetadata, right.pack_type)?.brand ?? ''
            : sortConfig.key === 'category'
              ? getSkuInfo(skuMetadata, right.pack_type)?.category ?? ''
              : getSortableValue(right, sortConfig.key);

      if (leftValue < rightValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }

      if (leftValue > rightValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }

      return 0;
    });
    return rows;
  }, [distributorMetadata, searchableRows, sortConfig, skuMetadata]);

  const totalPages = Math.max(1, Math.ceil(sortedRows.length / PAGE_SIZE));

  useEffect(() => {
    setCurrentPage(1);
  }, [selectedDistributor, selectedSku, selectedHorizon, selectedMonth, searchQuery, sortConfig]);

  const paginatedRows = useMemo(() => {
    const startIndex = (currentPage - 1) * PAGE_SIZE;
    return sortedRows.slice(startIndex, startIndex + PAGE_SIZE);
  }, [currentPage, sortedRows]);

  const hasData = filteredRecords.length > 0;

  const toggleSort = (key) => {
    setSortConfig((currentSort) => {
      if (currentSort.key === key) {
        return {
          key,
          direction: currentSort.direction === 'asc' ? 'desc' : 'asc',
        };
      }

      return {
        key,
        direction: 'asc',
      };
    });
  };

  const SortLabel = ({ columnKey, children, align = 'left' }) => {
    const isActive = sortConfig.key === columnKey;
    const Icon = isActive && sortConfig.direction === 'desc' ? ChevronDown : ChevronUp;

    return (
      <button
        type="button"
        onClick={() => toggleSort(columnKey)}
        className={`inline-flex items-center gap-1 font-medium ${align === 'right' ? 'ml-auto' : ''}`}
      >
        <span>{children}</span>
        <Icon
          className={`w-4 h-4 ${isActive ? 'text-brand-accent' : 'text-slate-300'}`}
        />
      </button>
    );
  };

  return (
    <div className="min-h-screen p-4 md:p-8 max-w-[1600px] mx-auto space-y-8 bg-slate-50 text-slate-900">
      <header className="flex flex-col xl:flex-row xl:items-end justify-between gap-5 pb-6 border-b border-slate-200">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-accent to-purple-500">
            Secondary Sales Forecast Dashboard
          </h1>
          <p className="text-brand-muted mt-1">
            Filter-driven KPIs, forecast confidence intervals, and business-friendly product analytics
          </p>
        </div>

        <button
          type="button"
          onClick={loadDashboard}
          disabled={refreshing}
          className="px-6 py-2 bg-brand-accent hover:bg-blue-600 disabled:opacity-60 disabled:cursor-not-allowed transition-colors rounded-lg font-medium text-sm text-white inline-flex items-center justify-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </header>

      <section className="bg-white p-5 rounded-xl border border-slate-200 shadow-md">
        <div className="flex items-center gap-2 mb-4">
          <Filter className="w-4 h-4 text-brand-accent" />
          <h2 className="font-semibold text-slate-900">Filters</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
            <Truck className="w-4 h-4 text-slate-500 shrink-0" />
            <select
              value={selectedDistributor}
              onChange={(event) => setSelectedDistributor(event.target.value)}
              className="w-full bg-transparent text-sm text-slate-900 focus:outline-none"
            >
              <option value="ALL">All Distributors</option>
              {distributorOptions.map((distributor) => (
                <option key={distributor.value} value={distributor.value}>
                  {distributor.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
            <Package className="w-4 h-4 text-slate-500 shrink-0" />
            <select
              value={selectedSku}
              onChange={(event) => setSelectedSku(event.target.value)}
              className="w-full bg-transparent text-sm text-slate-900 focus:outline-none"
            >
              <option value="ALL">All Products</option>
              {skuOptions.map((sku) => (
                <option key={sku.value} value={sku.value}>
                  {sku.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
            <TrendingUp className="w-4 h-4 text-slate-500 shrink-0" />
            <select
              value={selectedHorizon}
              onChange={(event) => setSelectedHorizon(event.target.value)}
              className="w-full bg-transparent text-sm text-slate-900 focus:outline-none"
            >
              <option value="ALL">All Horizons</option>
              {HORIZONS.map((horizon) => (
                <option key={horizon} value={horizon}>
                  {horizon}-Day
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
            <CalendarRange className="w-4 h-4 text-slate-500 shrink-0" />
            <select
              value={selectedMonth}
              onChange={(event) => setSelectedMonth(event.target.value)}
              className="w-full bg-transparent text-sm text-slate-900 focus:outline-none"
            >
              <option value="ALL">All Forecast Months</option>
              {monthOptions.map((month) => (
                <option key={month} value={month}>
                  {formatMonth(month)}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
            <Search className="w-4 h-4 text-slate-500 shrink-0" />
            <input
              type="text"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search product, brand, category"
              className="w-full bg-transparent text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
            />
          </div>
        </div>
      </section>

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-brand-accent">
          <Activity className="w-8 h-8 animate-spin mb-4" />
          <p>Loading Secondary Sales forecast data...</p>
        </div>
      )}

      {error && !loading && (
        <div className="flex items-start gap-3 bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg">
          <AlertCircle className="w-5 h-5 mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">Unable to load Secondary Sales forecast data</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      )}

      {!loading && !error && (
        <>
          <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {summaryCards.map((card) => {
              const Icon = card.icon;

              return (
                <div
                  key={card.title}
                  className="bg-white p-5 rounded-xl border border-[#E2E8F0] shadow-md relative overflow-hidden group hover:border-blue-400/50 transition-colors"
                >
                  <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/5 to-transparent rounded-bl-full -z-10 group-hover:from-blue-500/10 transition-colors" />
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold text-slate-700">{card.title}</h2>
                    <Icon className="w-5 h-5 text-brand-accent opacity-70" />
                  </div>
                  <p className="text-2xl font-bold text-slate-900 break-words">{card.value}</p>
                  <p className="text-sm text-slate-500 mt-2">{card.subtext}</p>
                </div>
              );
            })}
          </section>

          {!hasData && (
            <div className="text-center py-20 text-slate-500 bg-white rounded-xl border border-slate-200 shadow-md">
              <p>No Secondary Sales forecast data is available for the selected filters.</p>
            </div>
          )}

          {hasData && (
            <div className="space-y-8">
              <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-md">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
                  <div>
                    <h2 className="font-semibold text-slate-900">Monthly Forecast Trend</h2>
                    <p className="text-sm text-slate-500 mt-1">
                      Straight-line monthly totals by horizon to avoid implying interpolated behavior between forecast months
                    </p>
                  </div>
                  <p className="text-sm text-slate-500">
                    Forecast window:{' '}
                    {status?.forecast_date_range?.[0] && status?.forecast_date_range?.[1]
                      ? `${formatMonth(status.forecast_date_range[0])} to ${formatMonth(status.forecast_date_range[1])}`
                      : 'Not available'}
                  </p>
                </div>

                <div className="h-[420px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 10, right: 24, left: 16, bottom: 16 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="label" stroke="#475569" tick={{ fill: '#475569', fontSize: 12 }} />
                      <YAxis
                        stroke="#475569"
                        tick={{ fill: '#475569', fontSize: 12 }}
                        tickFormatter={formatQuantity}
                      />
                      <Tooltip
                        formatter={(value) => formatQuantity(value)}
                        labelFormatter={(label) => `Forecast Month: ${label}`}
                      />
                      <Legend />
                      <Line
                        type="linear"
                        connectNulls={false}
                        dataKey="qty30"
                        stroke="#2563eb"
                        strokeWidth={3}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                        name="30-Day Forecast"
                      />
                      <Line
                        type="linear"
                        connectNulls={false}
                        dataKey="qty60"
                        stroke="#10b981"
                        strokeWidth={3}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                        name="60-Day Forecast"
                      />
                      <Line
                        type="linear"
                        connectNulls={false}
                        dataKey="qty90"
                        stroke="#f59e0b"
                        strokeWidth={3}
                        dot={{ r: 4 }}
                        activeDot={{ r: 6 }}
                        name="90-Day Forecast"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="space-y-4">
                <div>
                  <h2 className="font-semibold text-slate-900">30/60/90-Day Forecast Comparison</h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Aggregated prediction totals with lower and upper confidence bounds from the current API response
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {comparisonCards.map((card) => (
                    <div
                      key={card.horizon}
                      className="bg-white p-5 rounded-xl border border-slate-200 shadow-md"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-slate-700">{card.horizon}-Day Forecast</h3>
                        <TrendingUp className="w-5 h-5 text-brand-accent opacity-70" />
                      </div>
                      <p className="text-sm text-slate-500 mb-1">Predicted Quantity</p>
                      <p className="text-3xl font-bold text-slate-900">{formatQuantity(card.totalQuantity)}</p>
                      <div className="grid grid-cols-2 gap-4 pt-4 mt-4 border-t border-slate-200">
                        <div>
                          <p className="text-xs text-slate-500 mb-1">Lower Bound (P10)</p>
                          <p className="font-medium text-red-500">{formatQuantity(card.lowerBound)}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-slate-500 mb-1">Upper Bound (P90)</p>
                          <p className="font-medium text-emerald-600">{formatQuantity(card.upperBound)}</p>
                        </div>
                      </div>
                      <p className="text-xs text-slate-500 mt-4">
                        Forecast months in view: {formatQuantity(card.uniqueMonths)}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-white p-6 rounded-xl border border-slate-200 shadow-md">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
                  <div>
                    <h2 className="font-semibold text-slate-900">Forecast Table</h2>
                    <p className="text-sm text-slate-500 mt-1">
                      Business-friendly product display from `dim_sku.parquet` with forecast rows from `secondary_forecasts.parquet`
                    </p>
                  </div>
                  <p className="text-sm text-slate-500">
                    Showing {formatQuantity(paginatedRows.length)} of {formatQuantity(sortedRows.length)} filtered rows
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-500">
                        <th className="text-left py-3 pr-4"><SortLabel columnKey="distributor_label">Distributor</SortLabel></th>
                        <th className="text-left py-3 pr-4"><SortLabel columnKey="product_label">Product</SortLabel></th>
                        {tableColumnAvailability.showBrand && (
                          <th className="text-left py-3 pr-4"><SortLabel columnKey="brand">Brand</SortLabel></th>
                        )}
                        {tableColumnAvailability.showCategory && (
                          <th className="text-left py-3 pr-4"><SortLabel columnKey="category">Category</SortLabel></th>
                        )}
                        <th className="text-left py-3 pr-4"><SortLabel columnKey="forecast_month">Forecast Month</SortLabel></th>
                        <th className="text-left py-3 pr-4"><SortLabel columnKey="horizon_days">Horizon</SortLabel></th>
                        <th className="text-right py-3 pr-4"><SortLabel columnKey="predicted_revenue" align="right">Predicted Quantity</SortLabel></th>
                        <th className="text-right py-3 pr-4"><SortLabel columnKey="P10" align="right">P10</SortLabel></th>
                        <th className="text-right py-3"><SortLabel columnKey="P90" align="right">P90</SortLabel></th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedRows.map((row, index) => {
                        const skuInfo = getSkuInfo(skuMetadata, row.pack_type);

                        return (
                          <tr
                            key={`${row.payer_code}-${row.pack_type}-${row.forecast_month}-${row.horizon_days}-${index}`}
                            className="border-b border-slate-100 hover:bg-slate-50"
                          >
                            <td className="py-3 pr-4 text-slate-900">
                              {getDistributorDisplayLabel(distributorMetadata, row.payer_code)}
                            </td>
                            <td className="py-3 pr-4 text-slate-900">{getSkuDisplayLabel(skuMetadata, row.pack_type)}</td>
                            {tableColumnAvailability.showBrand && (
                              <td className="py-3 pr-4 text-slate-600">{skuInfo?.brand || '—'}</td>
                            )}
                            {tableColumnAvailability.showCategory && (
                              <td className="py-3 pr-4 text-slate-600">{skuInfo?.category || '—'}</td>
                            )}
                            <td className="py-3 pr-4 text-slate-600">{formatMonth(row.forecast_month)}</td>
                            <td className="py-3 pr-4 text-slate-600">{row.horizon_days}-Day</td>
                            <td className="py-3 pr-4 text-right font-medium text-slate-900">
                              {formatQuantity(row.predicted_revenue)}
                            </td>
                            <td className="py-3 pr-4 text-right text-red-500">{formatQuantity(row.P10)}</td>
                            <td className="py-3 text-right text-emerald-600">{formatQuantity(row.P90)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mt-6 pt-4 border-t border-slate-200">
                  <p className="text-sm text-slate-500">
                    Page {currentPage} of {totalPages}
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                      disabled={currentPage === 1}
                      className="px-4 py-2 rounded-lg border border-slate-300 text-sm text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                    >
                      Previous
                    </button>
                    <button
                      type="button"
                      onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
                      disabled={currentPage === totalPages}
                      className="px-4 py-2 rounded-lg border border-slate-300 text-sm text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  );
}
