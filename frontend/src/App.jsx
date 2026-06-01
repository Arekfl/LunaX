import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { MapContainer, Rectangle, useMap, useMapEvents, WMSTileLayer } from "react-leaflet";

const GEO_BOUNDS = [
  [-90, -180],
  [90, 180],
];

const LON_MIN = GEO_BOUNDS[0][1];
const LON_MAX = GEO_BOUNDS[1][1];
const LAT_MIN = GEO_BOUNDS[0][0];
const LAT_MAX = GEO_BOUNDS[1][0];
const GEO_WIDTH = LON_MAX - LON_MIN;
const GEO_HEIGHT = LAT_MAX - LAT_MIN;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_DETECTION_STATUS = "to_verify";
const STATUS_COLOR_MAP = {
  confirmed: "#198754",
  to_verify: "#ffc107",
  rejected: "#dc3545",
};
const STATUS_BADGE_CLASS_MAP = {
  confirmed: "text-bg-success",
  to_verify: "text-bg-warning",
  rejected: "text-bg-danger",
};
const DETECTION_BBOX_PROXIMITY_THRESHOLD = 12;
const NO_DETECTIONS_FILTER = "no_detections";
const RESOLUTION_DESCRIPTION_MAP = {
  preview: "Szybki podglad, nizsza dokladnosc.",
  detail: "Zbalansowany tryb do codziennej analizy.",
  ultra: "Najwyzsza dokladnosc, najdluzszy czas analizy.",
};
const RESOLUTION_MPP_MAP = {
  preview: 59.22,
  detail: 15.79,
  ultra: 0.87,
};
const RESOLUTION_IMAGE_SIZE_MAP = {
  preview: 1024,
  detail: 1536,
  ultra: 2048,
};

const GRID_SIZE = 4;
const GRID_ROWS = GRID_SIZE;
const GRID_COLS = GRID_SIZE;

function buildGridCells(bounds, level) {
  const rows = GRID_ROWS;
  const cols = GRID_COLS;
  const [[latMin, lonMin], [latMax, lonMax]] = bounds;
  const cellHeight = (latMax - latMin) / rows;
  const cellWidth = (lonMax - lonMin) / cols;
  const segments = [];

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const yMin = latMin + row * cellHeight;
      const xMin = lonMin + col * cellWidth;
      const yMax = row === rows - 1 ? latMax : latMin + (row + 1) * cellHeight;
      const xMax = col === cols - 1 ? lonMax : lonMin + (col + 1) * cellWidth;

      segments.push({
        id: `level-${level}-segment-${row + 1}-${col + 1}`,
        level,
        row,
        col,
        bounds: [
          [yMin, xMin],
          [yMax, xMax],
        ],
      });
    }
  }

  return segments;
}

function boundsToCoords(bounds) {
  const [[yMin, xMin], [yMax, xMax]] = bounds;
  return { xMin, yMin, xMax, yMax };
}

function parseBBoxToMinMax(bbox) {
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [xMin, yMin, xMax, yMax] = bbox;
    return { xMin, yMin, xMax, yMax };
  }

  if (!bbox || typeof bbox !== "object") {
    return null;
  }

  if (
    typeof bbox.xMin === "number" &&
    typeof bbox.yMin === "number" &&
    typeof bbox.xMax === "number" &&
    typeof bbox.yMax === "number"
  ) {
    return {
      xMin: bbox.xMin,
      yMin: bbox.yMin,
      xMax: bbox.xMax,
      yMax: bbox.yMax,
    };
  }

  if (
    typeof bbox.x === "number" &&
    typeof bbox.y === "number" &&
    typeof bbox.width === "number" &&
    typeof bbox.height === "number"
  ) {
    return {
      xMin: bbox.x,
      yMin: bbox.y,
      xMax: bbox.x + bbox.width,
      yMax: bbox.y + bbox.height,
    };
  }

  return null;
}

function clampValue(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function isBoundsInsideImage(bounds) {
  if (!bounds) {
    return false;
  }

  const [[yMin, xMin], [yMax, xMax]] = bounds;

  return (
    Number.isFinite(yMin) &&
    Number.isFinite(xMin) &&
    Number.isFinite(yMax) &&
    Number.isFinite(xMax) &&
    yMin >= GEO_BOUNDS[0][0] &&
    xMin >= GEO_BOUNDS[0][1] &&
    yMax <= GEO_BOUNDS[1][0] &&
    xMax <= GEO_BOUNDS[1][1] &&
    yMax > yMin &&
    xMax > xMin
  );
}

function detectionToBounds(detection) {
  const bboxMinMax = parseBBoxToMinMax(detection?.bbox);
  if (!bboxMinMax) {
    return null;
  }

  const { xMin, yMin, xMax, yMax } = bboxMinMax;
  const bounds = [
    [yMin, xMin],
    [yMax, xMax],
  ];

  return isBoundsInsideImage(bounds) ? bounds : null;
}

function getDetectionCenter(detection) {
  const bboxMinMax = parseBBoxToMinMax(detection?.bbox);
  if (!bboxMinMax) {
    return null;
  }

  return {
    lon: (bboxMinMax.xMin + bboxMinMax.xMax) / 2,
    lat: (bboxMinMax.yMin + bboxMinMax.yMax) / 2,
  };
}

function resolveAnalysisImageForDetection(detection, analysisImageList) {
  if (!detection || !Array.isArray(analysisImageList) || analysisImageList.length === 0) {
    return null;
  }

  const targetAnalysisId = String(detection.analysis_id || "").trim();
  if (!targetAnalysisId) {
    return null;
  }

  const sameAnalysisImages = analysisImageList.filter(
    (image) => String(image.analysis_id || "").trim() === targetAnalysisId
  );
  if (sameAnalysisImages.length === 0) {
    return null;
  }

  const withDetectionsImages = sameAnalysisImages.filter(
    (image) => String(image.status || "").trim().toLowerCase() !== NO_DETECTIONS_FILTER
  );
  const candidates = withDetectionsImages.length > 0 ? withDetectionsImages : sameAnalysisImages;

  const center = getDetectionCenter(detection);
  if (center) {
    let bestImage = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const image of candidates) {
      const lat = Number(image?.lat);
      const lon = Number(image?.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        continue;
      }

      const squaredDistance = (lat - center.lat) ** 2 + (lon - center.lon) ** 2;
      if (squaredDistance < bestDistance) {
        bestDistance = squaredDistance;
        bestImage = image;
      }
    }

    if (bestImage) {
      return bestImage;
    }
  }

  return [...candidates].sort((leftImage, rightImage) => {
    const leftTimestamp = String(leftImage?.timestamp || "");
    const rightTimestamp = String(rightImage?.timestamp || "");
    return rightTimestamp.localeCompare(leftTimestamp);
  })[0] ?? null;
}

function normalizeBBoxToPixelMinMax(bbox, originalWidth, originalHeight) {
  const parsedBBox = parseBBoxToMinMax(bbox);
  if (!parsedBBox) {
    return null;
  }

  const sourceValues = [parsedBBox.xMin, parsedBBox.yMin, parsedBBox.xMax, parsedBBox.yMax];
  if (!sourceValues.every((value) => Number.isFinite(value))) {
    return null;
  }

  const isYOLO01 =
    parsedBBox.xMin >= 0 &&
    parsedBBox.yMin >= 0 &&
    parsedBBox.xMax >= 0 &&
    parsedBBox.yMax >= 0 &&
    parsedBBox.xMin <= 1 &&
    parsedBBox.yMin <= 1 &&
    parsedBBox.xMax <= 1 &&
    parsedBBox.yMax <= 1;

  const xMinRaw = isYOLO01 ? parsedBBox.xMin * originalWidth : parsedBBox.xMin;
  const yMinRaw = isYOLO01 ? parsedBBox.yMin * originalHeight : parsedBBox.yMin;
  const xMaxRaw = isYOLO01 ? parsedBBox.xMax * originalWidth : parsedBBox.xMax;
  const yMaxRaw = isYOLO01 ? parsedBBox.yMax * originalHeight : parsedBBox.yMax;

  const xMin = clampValue(Math.min(xMinRaw, xMaxRaw), 0, originalWidth);
  const yMin = clampValue(Math.min(yMinRaw, yMaxRaw), 0, originalHeight);
  const xMax = clampValue(Math.max(xMinRaw, xMaxRaw), 0, originalWidth);
  const yMax = clampValue(Math.max(yMinRaw, yMaxRaw), 0, originalHeight);

  if (xMax <= xMin || yMax <= yMin) {
    return null;
  }

  return [xMin, yMin, xMax, yMax];
}

function getDetectionOverlayRect(detection, image, imageMetrics) {
  const displayWidth = Number(imageMetrics?.displayWidth);
  const displayHeight = Number(imageMetrics?.displayHeight);
  const naturalWidth = Number(imageMetrics?.naturalWidth);
  const naturalHeight = Number(imageMetrics?.naturalHeight);

  const resolution = String(image?.resolution || detection?.resolution || "").trim();
  const fallbackSize = Number(RESOLUTION_IMAGE_SIZE_MAP[resolution] || 0);

  const originalWidth = naturalWidth > 0 ? naturalWidth : fallbackSize;
  const originalHeight = naturalHeight > 0 ? naturalHeight : fallbackSize;

  if (displayWidth <= 0 || displayHeight <= 0 || originalWidth <= 0 || originalHeight <= 0) {
    return null;
  }

  const pixelBBox = normalizeBBoxToPixelMinMax(detection?.bbox, originalWidth, originalHeight);
  if (!pixelBBox) {
    return null;
  }

  const [xMin, yMin, xMax, yMax] = pixelBBox;
  const scaleX = displayWidth / originalWidth;
  const scaleY = displayHeight / originalHeight;

  const left = xMin * scaleX;
  const top = yMin * scaleY;
  const width = (xMax - xMin) * scaleX;
  const height = (yMax - yMin) * scaleY;

  const isOutsideView =
    width <= 0 ||
    height <= 0 ||
    left >= displayWidth ||
    top >= displayHeight ||
    left + width <= 0 ||
    top + height <= 0;

  if (isOutsideView) {
    console.log("BBox poza widokiem podgladu", {
      detectionId: detection?.detection_id,
      pixelBBox: [xMin, yMin, xMax, yMax],
      displayWidth,
      displayHeight,
      originalWidth,
      originalHeight,
      scaleX,
      scaleY,
      left,
      top,
      width,
      height,
    });
    return null;
  }

  return {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`,
  };
}

function areBBoxesClose(leftBBox, rightBBox, threshold) {
  const left = parseBBoxToMinMax(leftBBox);
  const right = parseBBoxToMinMax(rightBBox);

  if (!left || !right) {
    return false;
  }

  return (
    Math.abs(left.xMin - right.xMin) < threshold &&
    Math.abs(left.yMin - right.yMin) < threshold &&
    Math.abs(left.xMax - right.xMax) < threshold &&
    Math.abs(left.yMax - right.yMax) < threshold
  );
}

function deduplicateDetectionsByProximity(detectionList, threshold) {
  const deduplicated = [];

  for (const detection of detectionList) {
    const matchIndex = deduplicated.findIndex((existing) => {
      if (existing.class !== detection.class) {
        return false;
      }

      return areBBoxesClose(existing.bbox, detection.bbox, threshold);
    });

    if (matchIndex === -1) {
      deduplicated.push(detection);
      continue;
    }

    const existing = deduplicated[matchIndex];
    if (Number(detection.confidence) > Number(existing.confidence)) {
      deduplicated[matchIndex] = detection;
    }
  }

  return deduplicated;
}

function detectionMatchesSelectedTags(detection, selectedTags, tagMatchMode) {
  const normalizedSelectedTags = normalizeDetectionTags(selectedTags);
  if (normalizedSelectedTags.length === 0) {
    return true;
  }

  const detectionTags = normalizeDetectionTags(detection?.tags);
  if (detectionTags.length === 0) {
    return false;
  }

  if (tagMatchMode === "and") {
    return normalizedSelectedTags.every((tag) => detectionTags.includes(tag));
  }

  return normalizedSelectedTags.some((tag) => detectionTags.includes(tag));
}

function getDisplayDetectionsForStatus(detectionList, status, options = {}) {
  const {
    requireValidBounds = true,
    selectedTags = [],
    tagMatchMode = "or",
  } = options;
  const statusMatched = detectionList.filter((detection) => detection.status === status);
  const tagsMatched = statusMatched.filter((detection) =>
    detectionMatchesSelectedTags(detection, selectedTags, tagMatchMode)
  );
  const scopeMatched = requireValidBounds
    ? tagsMatched.filter((detection) => detectionToBounds(detection))
    : tagsMatched;

  return deduplicateDetectionsByProximity(scopeMatched, DETECTION_BBOX_PROXIMITY_THRESHOLD);
}

function getDetectionConfidenceValue(detection) {
  const numericConfidence = Number(detection?.confidence);
  return Number.isFinite(numericConfidence) ? numericConfidence : null;
}

function getDetectionTimestampValue(detection) {
  const timestampRaw = typeof detection?.timestamp === "string" ? detection.timestamp.trim() : "";
  if (!timestampRaw) {
    return null;
  }

  const parsedTimestamp = Date.parse(timestampRaw);
  return Number.isFinite(parsedTimestamp) ? parsedTimestamp : null;
}

function compareNullableNumbers(leftValue, rightValue, sortOrder) {
  const leftIsValid = Number.isFinite(leftValue);
  const rightIsValid = Number.isFinite(rightValue);

  if (!leftIsValid && !rightIsValid) {
    return 0;
  }

  if (!leftIsValid) {
    return 1;
  }

  if (!rightIsValid) {
    return -1;
  }

  return sortOrder === "asc" ? leftValue - rightValue : rightValue - leftValue;
}

function sortDetectionList(detectionList, sortBy, sortOrder) {
  const normalizedSortBy = sortBy === "confidence" ? "confidence" : "data";
  const normalizedSortOrder = sortOrder === "asc" ? "asc" : "desc";
  const indexedDetections = detectionList.map((detection, index) => ({ detection, index }));

  indexedDetections.sort((leftItem, rightItem) => {
    if (normalizedSortBy === "confidence") {
      const confidenceDelta = compareNullableNumbers(
        getDetectionConfidenceValue(leftItem.detection),
        getDetectionConfidenceValue(rightItem.detection),
        normalizedSortOrder
      );
      if (confidenceDelta !== 0) {
        return confidenceDelta;
      }

      const timestampDelta = compareNullableNumbers(
        getDetectionTimestampValue(leftItem.detection),
        getDetectionTimestampValue(rightItem.detection),
        "desc"
      );
      if (timestampDelta !== 0) {
        return timestampDelta;
      }
    } else {
      const timestampDelta = compareNullableNumbers(
        getDetectionTimestampValue(leftItem.detection),
        getDetectionTimestampValue(rightItem.detection),
        normalizedSortOrder
      );
      if (timestampDelta !== 0) {
        return timestampDelta;
      }

      const confidenceDelta = compareNullableNumbers(
        getDetectionConfidenceValue(leftItem.detection),
        getDetectionConfidenceValue(rightItem.detection),
        "desc"
      );
      if (confidenceDelta !== 0) {
        return confidenceDelta;
      }
    }

    return leftItem.index - rightItem.index;
  });

  return indexedDetections.map((item) => item.detection);
}

function isNoCoverageErrorMessage(message) {
  if (typeof message !== "string") {
    return false;
  }

  const normalizedMessage = message.toLowerCase();
  return (
    normalizedMessage.includes("no valid imagery") ||
    normalizedMessage.includes("fully transparent") ||
    normalizedMessage.includes("flat (single-value image)")
  );
}

function getDetectionUniqueId(detection) {
  const { analysis_id: analysisId, detection_id: detectionId, bbox } = detection;
  return [analysisId ?? "no-analysis", detectionId, bbox.x, bbox.y, bbox.width, bbox.height].join("|");
}

function isSameDetection(leftDetection, rightDetection) {
  if (!leftDetection || !rightDetection) {
    return false;
  }

  return getDetectionUniqueId(leftDetection) === getDetectionUniqueId(rightDetection);
}

function resolveDetectionStatus(detection, statusMap) {
  return statusMap[detection.detection_id] ?? detection.status ?? DEFAULT_DETECTION_STATUS;
}

function normalizeDetectionTags(tags) {
  if (!Array.isArray(tags)) {
    return [];
  }

  const normalizedTags = tags
    .map((tag) => String(tag).trim())
    .filter((tag) => tag.length > 0);

  return [...new Set(normalizedTags)];
}

function applyStatusesToDetections(detectionList, statusMap) {
  return detectionList.map((detection) => ({
    ...detection,
    status: statusMap[detection.detection_id] ?? detection.status ?? DEFAULT_DETECTION_STATUS,
    tags: normalizeDetectionTags(detection.tags),
  }));
}

function getDetectionStatusCounts(detectionList) {
  const counts = {
    confirmed: 0,
    to_verify: 0,
    approved: 0,
  };

  for (const detection of detectionList) {
    const normalizedStatus =
      typeof detection?.status === "string" ? detection.status.trim().toLowerCase() : "";

    if (normalizedStatus === "confirmed") {
      counts.confirmed += 1;
      continue;
    }

    if (normalizedStatus === "to_verify") {
      counts.to_verify += 1;
      continue;
    }

    if (normalizedStatus === "approved") {
      counts.approved += 1;
    }
  }

  return counts;
}

function formatDetectionStatusSummary(detectionList) {
  const counts = getDetectionStatusCounts(detectionList);
  return (
    `Statusy: confirmed ${counts.confirmed}, ` +
    `to_verify ${counts.to_verify}, approved ${counts.approved}.`
  );
}

function getStatusColor(status) {
  return STATUS_COLOR_MAP[status] ?? STATUS_COLOR_MAP[DEFAULT_DETECTION_STATUS];
}

function getStatusBadgeClass(status) {
  return STATUS_BADGE_CLASS_MAP[status] ?? "text-bg-secondary";
}

function getFileNameFromPath(filePath) {
  if (!filePath) {
    return "";
  }

  const normalizedPath = String(filePath).replaceAll("\\", "/");
  const parts = normalizedPath.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

function formatCoordinate(value, digits = 2) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "-";
  }

  return numericValue.toFixed(digits);
}

function escapeCsvCell(value) {
  const stringValue = String(value ?? "");
  const escapedValue = stringValue.replaceAll('"', '""');
  return `"${escapedValue}"`;
}

function triggerTextDownload(content, fileName, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const downloadUrl = URL.createObjectURL(blob);
  const linkElement = document.createElement("a");
  linkElement.href = downloadUrl;
  linkElement.download = fileName;
  document.body.appendChild(linkElement);
  linkElement.click();
  document.body.removeChild(linkElement);
  URL.revokeObjectURL(downloadUrl);
}

function getAnalysisImageUrl(imageId) {
  if (!imageId) {
    return null;
  }

  return `${API_BASE_URL}/analysis-images/image/${encodeURIComponent(imageId)}`;
}

function getDetectionPreviewUrl(detection) {
  const bboxMinMax = parseBBoxToMinMax(detection?.bbox);
  if (!bboxMinMax) {
    return null;
  }

  const { xMin, yMin, xMax, yMax } = bboxMinMax;
  const params = new URLSearchParams({
    map: "/maps/earth/moon_simp_cyl.map",
    SERVICE: "WMS",
    VERSION: "1.1.1",
    REQUEST: "GetMap",
    LAYERS: "KaguyaTC_Ortho",
    STYLES: "",
    FORMAT: "image/png",
    SRS: "EPSG:4326",
    BBOX: `${xMin},${yMin},${xMax},${yMax}`,
    WIDTH: "512",
    HEIGHT: "512",
  });

  return `https://planetarymaps.usgs.gov/cgi-bin/mapserv?${params.toString()}`;
}

function FitBoundsOnChange({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [0, 0], animate: false });
    }
  }, [map, bounds]);

  return null;
}

function HomeControl({ onHomeClick }) {
  const map = useMap();

  useEffect(() => {
    const homeControl = L.control({ position: "topleft" });

    homeControl.onAdd = () => {
      const container = L.DomUtil.create("div", "leaflet-bar");
      const button = L.DomUtil.create("a", "leaflet-control-home", container);

      button.href = "#";
      button.title = "Pokaz cala mape";
      button.innerHTML = "&#8962;";

      L.DomEvent.disableClickPropagation(container);
      L.DomEvent.on(button, "click", (event) => {
        L.DomEvent.preventDefault(event);
        onHomeClick();
      });

      return container;
    };

    homeControl.addTo(map);

    return () => {
      homeControl.remove();
    };
  }, [map, onHomeClick]);

  return null;
}

function ZoomOutLevelControl({ currentLevel, onStepOut, suppressZoomOutRef }) {
  const lastZoomRef = useRef(null);
  const map = useMapEvents({
    zoomend() {
      const currentZoom = map.getZoom();
      const previousZoom = lastZoomRef.current;

      if (suppressZoomOutRef.current > 0) {
        suppressZoomOutRef.current -= 1;
        lastZoomRef.current = currentZoom;
        return;
      }

      if (typeof previousZoom === "number" && currentZoom < previousZoom && currentLevel > 0) {
        onStepOut();
      }

      lastZoomRef.current = currentZoom;
    },
  });

  useEffect(() => {
    lastZoomRef.current = map.getZoom();
  }, [map]);

  return null;
}

export default function App() {
  const mapRef = useRef(null);
  const detectionListRef = useRef(null);
  const detectionItemRefs = useRef(new Map());
  const masterDetectionsCheckboxRef = useRef(null);
  const masterNoDetectionsCheckboxRef = useRef(null);
  const tagFilterDropdownRef = useRef(null);
  const detectionPreviewImageRef = useRef(null);
  const suppressZoomOutRef = useRef(0);
  const [currentLevel, setCurrentLevel] = useState(0);
  const [isLevelLocked, setIsLevelLocked] = useState(false);
  const [selectedBBox, setSelectedBBox] = useState(GEO_BOUNDS);
  const [bboxHistory, setBBoxHistory] = useState([GEO_BOUNDS]);
  const [gridCells, setGridCells] = useState(() => buildGridCells(GEO_BOUNDS, 0));
  const [detections, setDetections] = useState([]);
  const [analysisImages, setAnalysisImages] = useState([]);
  const [selectedNoDetectionImage, setSelectedNoDetectionImage] = useState(null);
  const [expandedNoDetectionImageId, setExpandedNoDetectionImageId] = useState(null);
  const [selectedNoDetectionImageIds, setSelectedNoDetectionImageIds] = useState([]);
  const [noDetectionBulkTagDraft, setNoDetectionBulkTagDraft] = useState("");
  const [isNoDetectionBulkTagging, setIsNoDetectionBulkTagging] = useState(false);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [isLoadingDetections, setIsLoadingDetections] = useState(false);
  const [analysisOverlayBounds, setAnalysisOverlayBounds] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [showBboxes, setShowBboxes] = useState(true);
  const [viewMode, setViewMode] = useState("map");
  const [resolutionMode, setResolutionMode] = useState("detail");
  const [numSamples, setNumSamples] = useState(5);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [storedStatuses, setStoredStatuses] = useState({});
  const [statusFilter, setStatusFilter] = useState("to_verify");
  const [detectionSortBy, setDetectionSortBy] = useState("confidence");
  const [detectionSortOrder, setDetectionSortOrder] = useState("desc");
  const [selectedTagFilters, setSelectedTagFilters] = useState([]);
  const [tagFilterMode, setTagFilterMode] = useState("or");
  const [isTagFilterDropdownOpen, setIsTagFilterDropdownOpen] = useState(false);
  const [hoveredSegmentId, setHoveredSegmentId] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [expandedDetectionId, setExpandedDetectionId] = useState(null);
  const [hoveredDetectionId, setHoveredDetectionId] = useState(null);
  const [inputComment, setInputComment] = useState("");
  const [tagDrafts, setTagDrafts] = useState({});
  const [editingDetectionId, setEditingDetectionId] = useState(null);
  const [selectedIds, setSelectedIds] = useState([]);
  const [bulkTagDraft, setBulkTagDraft] = useState("");
  const [isBulkTagging, setIsBulkTagging] = useState(false);
  const [exportFormat, setExportFormat] = useState("json");
  const [deleteModal, setDeleteModal] = useState({
    targetType: null,
    targetIds: [],
    deleteImages: false,
    isDeleting: false,
    missingImageWarning: false,
  });
  const [detectionPreviewModal, setDetectionPreviewModal] = useState({
    isOpen: false,
    detection: null,
    image: null,
    showBBoxPreview: true,
  });
  const [detectionPreviewImageMetrics, setDetectionPreviewImageMetrics] = useState({
    naturalWidth: 0,
    naturalHeight: 0,
    displayWidth: 0,
    displayHeight: 0,
  });
  const [focusBounds, setFocusBounds] = useState(GEO_BOUNDS);
  const [chosenMessage, setChosenMessage] = useState("");
  const [manualCoords, setManualCoords] = useState({
    xMin: String(LON_MIN),
    yMin: String(LAT_MIN),
    xMax: String(LON_MAX),
    yMax: String(LAT_MAX),
  });

  const selectedCoords = selectedSegment ? boundsToCoords(selectedSegment.bounds) : null;
  const isAnalysisLoading = isLoadingDetections || analysisStatus === "loading";
  const isNoDetectionsFilterSelected = statusFilter === NO_DETECTIONS_FILTER;
  const isNoCoverageChosenMessage =
    typeof chosenMessage === "string" &&
    chosenMessage.startsWith("Brak pokrycia danych dla tej warstwy i obszaru.");
  const showDetectedResultsInGallery = useCallback(
    (analysisLabel, analysisId, detectionCount, statusSummary) => {
      setViewMode("gallery");
      setChosenMessage(
        `${analysisLabel} ${analysisId} zakonczona. Znaleziono ${detectionCount} detekcji – pokazuję wyniki w galerii. ${statusSummary}`
      );
    },
    []
  );
  const updateDetectionPreviewImageMetrics = useCallback(() => {
    const imageNode = detectionPreviewImageRef.current;
    if (!imageNode) {
      return;
    }

    setDetectionPreviewImageMetrics({
      naturalWidth: Number(imageNode.naturalWidth) || 0,
      naturalHeight: Number(imageNode.naturalHeight) || 0,
      displayWidth: Number(imageNode.clientWidth) || 0,
      displayHeight: Number(imageNode.clientHeight) || 0,
    });
  }, []);
  const detectionPreviewOverlayRect = useMemo(
    () =>
      getDetectionOverlayRect(
        detectionPreviewModal.detection,
        detectionPreviewModal.image,
        detectionPreviewImageMetrics
      ),
    [detectionPreviewModal.detection, detectionPreviewModal.image, detectionPreviewImageMetrics]
  );

  useEffect(() => {
    setGridCells(buildGridCells(selectedBBox, currentLevel));
    setHoveredSegmentId(null);
  }, [selectedBBox, currentLevel]);

  const statusResolvedDetections = useMemo(
    () =>
      detections.map((detection) => ({
        ...detection,
        status: resolveDetectionStatus(detection, storedStatuses),
        tags: normalizeDetectionTags(detection.tags),
      })),
    [detections, storedStatuses]
  );

  const detectionsForCurrentView = useMemo(
    () =>
      getDisplayDetectionsForStatus(statusResolvedDetections, statusFilter, {
        requireValidBounds: false,
        selectedTags: selectedTagFilters,
        tagMatchMode: tagFilterMode,
      }),
    [statusResolvedDetections, statusFilter, selectedTagFilters, tagFilterMode]
  );

  const mapDetections = useMemo(
    () =>
      getDisplayDetectionsForStatus(statusResolvedDetections, statusFilter, {
        selectedTags: selectedTagFilters,
        tagMatchMode: tagFilterMode,
      }),
    [statusResolvedDetections, statusFilter, selectedTagFilters, tagFilterMode]
  );

  const availableDetectionTags = useMemo(() => {
    if (isNoDetectionsFilterSelected) {
      return [];
    }

    const tagsSet = new Set();
    for (const detection of statusResolvedDetections) {
      if (detection.status !== statusFilter) {
        continue;
      }

      for (const tag of normalizeDetectionTags(detection.tags)) {
        tagsSet.add(tag);
      }
    }

    return Array.from(tagsSet).sort((leftTag, rightTag) => leftTag.localeCompare(rightTag));
  }, [statusResolvedDetections, statusFilter, isNoDetectionsFilterSelected]);

  const visibleAnalysisImages = useMemo(
    () =>
      analysisImages.filter(
        (image) =>
          (typeof image.status === "string" ? image.status : NO_DETECTIONS_FILTER) ===
          NO_DETECTIONS_FILTER
      ),
    [analysisImages]
  );

  const hasConfidenceSortData = useMemo(
    () => detectionsForCurrentView.some((detection) => Number.isFinite(Number(detection?.confidence))),
    [detectionsForCurrentView]
  );

  const hasDateSortDataInDetections = useMemo(
    () => detectionsForCurrentView.some((detection) => Number.isFinite(getDetectionTimestampValue(detection))),
    [detectionsForCurrentView]
  );

  const hasDateSortDataInNoDetections = useMemo(
    () => visibleAnalysisImages.some((image) => Number.isFinite(getDetectionTimestampValue(image))),
    [visibleAnalysisImages]
  );

  const availableQuickSortFields = useMemo(() => {
    const fields = [];

    if (!isNoDetectionsFilterSelected && hasConfidenceSortData) {
      fields.push({ key: "confidence", label: "confidence" });
    }

    const hasDateData = isNoDetectionsFilterSelected
      ? hasDateSortDataInNoDetections
      : hasDateSortDataInDetections;
    if (hasDateData) {
      fields.push({ key: "data", label: "date" });
    }

    return fields;
  }, [
    isNoDetectionsFilterSelected,
    hasConfidenceSortData,
    hasDateSortDataInDetections,
    hasDateSortDataInNoDetections,
  ]);

  const effectiveSortBy =
    availableQuickSortFields.some((field) => field.key === detectionSortBy)
      ? detectionSortBy
      : availableQuickSortFields[0]?.key ?? "data";
  const effectiveSortOrder = detectionSortOrder === "asc" ? "asc" : "desc";

  const filteredDetections = useMemo(
    () => sortDetectionList(detectionsForCurrentView, effectiveSortBy, effectiveSortOrder),
    [detectionsForCurrentView, effectiveSortBy, effectiveSortOrder]
  );

  const sortedNoDetectionImages = useMemo(
    () => sortDetectionList(visibleAnalysisImages, effectiveSortBy, effectiveSortOrder),
    [visibleAnalysisImages, effectiveSortBy, effectiveSortOrder]
  );

  const backendSortBy = effectiveSortBy === "confidence" ? "confidence" : "data";
  const backendSortOrder = effectiveSortOrder === "asc" ? "asc" : "desc";

  const handleSortFieldClick = useCallback((fieldKey) => {
    if (detectionSortBy === fieldKey) {
      setDetectionSortOrder((previousSortOrder) =>
        previousSortOrder === "asc" ? "desc" : "asc"
      );
      return;
    }

    setDetectionSortBy(fieldKey);
    setDetectionSortOrder("desc");
  }, [detectionSortBy]);

  const detectionSectionCount = isNoDetectionsFilterSelected
    ? sortedNoDetectionImages.length
    : filteredDetections.length;

  const selectableDetectionIds = useMemo(
    () =>
      filteredDetections
        .map((detection) => String(detection?.detection_id || "").trim())
        .filter(Boolean),
    [filteredDetections]
  );

  const selectableNoDetectionImageIds = useMemo(
    () =>
      sortedNoDetectionImages
        .map((image) => String(image?.image_id || "").trim())
        .filter(Boolean),
    [sortedNoDetectionImages]
  );

  const selectedVisibleDetectionsCount = useMemo(() => {
    if (selectableDetectionIds.length === 0) {
      return 0;
    }

    const selectedSet = new Set(
      selectedIds.map((detectionId) => String(detectionId || "").trim()).filter(Boolean)
    );
    return selectableDetectionIds.filter((detectionId) => selectedSet.has(detectionId)).length;
  }, [selectedIds, selectableDetectionIds]);

  const selectedVisibleNoDetectionsCount = useMemo(() => {
    if (selectableNoDetectionImageIds.length === 0) {
      return 0;
    }

    const selectedSet = new Set(
      selectedNoDetectionImageIds
        .map((imageId) => String(imageId || "").trim())
        .filter(Boolean)
    );
    return selectableNoDetectionImageIds.filter((imageId) => selectedSet.has(imageId)).length;
  }, [selectedNoDetectionImageIds, selectableNoDetectionImageIds]);

  const areAllVisibleDetectionsSelected =
    selectableDetectionIds.length > 0 &&
    selectedVisibleDetectionsCount === selectableDetectionIds.length;
  const areSomeVisibleDetectionsSelected =
    selectedVisibleDetectionsCount > 0 && !areAllVisibleDetectionsSelected;

  const areAllVisibleNoDetectionsSelected =
    selectableNoDetectionImageIds.length > 0 &&
    selectedVisibleNoDetectionsCount === selectableNoDetectionImageIds.length;
  const areSomeVisibleNoDetectionsSelected =
    selectedVisibleNoDetectionsCount > 0 && !areAllVisibleNoDetectionsSelected;

  const fetchDetectionStatuses = useCallback(async () => {
    const statusesResponse = await fetch(`${API_BASE_URL}/detections/statuses`);

    if (!statusesResponse.ok) {
      throw new Error(`Detection statuses HTTP ${statusesResponse.status}`);
    }

    const statusesPayload = await statusesResponse.json();
    const statusMap =
      statusesPayload && typeof statusesPayload === "object" && !Array.isArray(statusesPayload)
        ? statusesPayload
        : {};

    setStoredStatuses(statusMap);
    return statusMap;
  }, []);

  const fetchDetectionsAndStatuses = useCallback(async () => {
    const queryParams = new URLSearchParams({
      sortBy: backendSortBy,
      sortOrder: backendSortOrder,
    });
    const [detectionsResponse, statusMap] = await Promise.all([
      fetch(`${API_BASE_URL}/detections/query?${queryParams.toString()}`),
      fetchDetectionStatuses(),
    ]);

    if (!detectionsResponse.ok) {
      throw new Error(`Detections query HTTP ${detectionsResponse.status}`);
    }

    const detectionsPayload = await detectionsResponse.json();

    const queriedDetections = Array.isArray(detectionsPayload) ? detectionsPayload : [];

    const mergedDetections = applyStatusesToDetections(queriedDetections, statusMap);
    setDetections(mergedDetections);

    return mergedDetections;
  }, [fetchDetectionStatuses, backendSortBy, backendSortOrder]);

  useEffect(() => {
    if (availableQuickSortFields.length === 0) {
      return;
    }

    const isCurrentFieldAvailable = availableQuickSortFields.some(
      (field) => field.key === detectionSortBy
    );
    if (isCurrentFieldAvailable) {
      return;
    }

    setDetectionSortBy(availableQuickSortFields[0].key);
    setDetectionSortOrder("desc");
  }, [availableQuickSortFields, detectionSortBy]);

  const fetchAnalysisImages = useCallback(async () => {
    const response = await fetch(`${API_BASE_URL}/analysis-images/query`);

    if (!response.ok) {
      throw new Error(`Analysis-images query HTTP ${response.status}`);
    }

    const payload = await response.json();
    const images = Array.isArray(payload) ? payload : [];
    setAnalysisImages(images);
    return images;
  }, []);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        await Promise.all([fetchDetectionsAndStatuses(), fetchAnalysisImages()]);
      } catch (error) {
        console.warn("Nie udalo sie pobrac danych poczatkowych:", error);
      }
    };

    loadInitialData();
  }, [fetchDetectionsAndStatuses, fetchAnalysisImages]);

  useEffect(() => {
    if (!selectedDetection) {
      return;
    }

    const isSelectedVisible = filteredDetections.some(
      (detection) => isSameDetection(detection, selectedDetection)
    );

    if (!isSelectedVisible) {
      setSelectedDetection(null);
    }
  }, [filteredDetections, selectedDetection]);

  useEffect(() => {
    if (!detectionPreviewModal.isOpen || !detectionPreviewModal.detection) {
      return;
    }

    const previewDetectionId = String(detectionPreviewModal.detection.detection_id || "");
    const stillExists = detections.some(
      (detection) => String(detection.detection_id || "") === previewDetectionId
    );

    if (!stillExists) {
      setDetectionPreviewModal((prevModal) => ({ ...prevModal, isOpen: false }));
    }
  }, [detectionPreviewModal, detections]);

  useEffect(() => {
    if (!selectedDetection) {
      return;
    }

    const selectedIndex = filteredDetections.findIndex((detection) =>
      isSameDetection(detection, selectedDetection)
    );
    if (selectedIndex < 0) {
      return;
    }

    const selectedRenderKey = `${getDetectionUniqueId(filteredDetections[selectedIndex])}|${selectedIndex}`;
    const selectedItem = detectionItemRefs.current.get(selectedRenderKey);
    const container = detectionListRef.current;
    if (!selectedItem || !container) {
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const itemRect = selectedItem.getBoundingClientRect();
    const padding = 8;

    if (itemRect.top < containerRect.top) {
      container.scrollTop -= containerRect.top - itemRect.top + padding;
      return;
    }

    if (itemRect.bottom > containerRect.bottom) {
      container.scrollTop += itemRect.bottom - containerRect.bottom + padding;
    }
  }, [filteredDetections, selectedDetection]);

  useEffect(() => {
    if (!hoveredDetectionId) {
      return;
    }

    const isHoveredVisible = filteredDetections.some(
      (detection) => getDetectionUniqueId(detection) === hoveredDetectionId
    );

    if (!isHoveredVisible) {
      setHoveredDetectionId(null);
    }
  }, [filteredDetections, hoveredDetectionId]);

  useEffect(() => {
    if (!editingDetectionId) {
      return;
    }

    const isEditingVisible = filteredDetections.some(
      (detection) => detection.detection_id === editingDetectionId
    );

    if (!isEditingVisible) {
      setEditingDetectionId(null);
      setInputComment("");
    }
  }, [filteredDetections, editingDetectionId]);

  useEffect(() => {
    if (!expandedDetectionId) {
      return;
    }

    const stillVisible = filteredDetections.some(
      (detection) => detection.detection_id === expandedDetectionId
    );

    if (!stillVisible) {
      setExpandedDetectionId(null);
      if (editingDetectionId === expandedDetectionId) {
        setEditingDetectionId(null);
        setInputComment("");
      }
    }
  }, [filteredDetections, expandedDetectionId, editingDetectionId]);

  useEffect(() => {
    const availableIds = new Set(detections.map((detection) => detection.detection_id));
    setSelectedIds((prevSelectedIds) =>
      prevSelectedIds.filter((detectionId) => availableIds.has(detectionId))
    );
    setTagDrafts((prevTagDrafts) => {
      const nextDrafts = { ...prevTagDrafts };
      for (const detectionId of Object.keys(nextDrafts)) {
        if (!availableIds.has(detectionId)) {
          delete nextDrafts[detectionId];
        }
      }
      return nextDrafts;
    });
  }, [detections]);

  useEffect(() => {
    if (!detectionPreviewModal.isOpen) {
      return;
    }

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setDetectionPreviewModal((prevModal) => ({ ...prevModal, isOpen: false }));
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [detectionPreviewModal.isOpen]);

  useEffect(() => {
    if (!detectionPreviewModal.isOpen) {
      return;
    }

    updateDetectionPreviewImageMetrics();

    const handleResize = () => {
      updateDetectionPreviewImageMetrics();
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [detectionPreviewModal.isOpen, updateDetectionPreviewImageMetrics]);

  useEffect(() => {
    const availableTagsSet = new Set(availableDetectionTags);
    setSelectedTagFilters((prevSelectedTags) =>
      prevSelectedTags.filter((tag) => availableTagsSet.has(tag))
    );
  }, [availableDetectionTags]);

  useEffect(() => {
    if (isNoDetectionsFilterSelected) {
      setIsTagFilterDropdownOpen(false);
    }
  }, [isNoDetectionsFilterSelected]);

  useEffect(() => {
    if (!isTagFilterDropdownOpen) {
      return;
    }

    const handleDocumentMouseDown = (event) => {
      const dropdownNode = tagFilterDropdownRef.current;
      if (!dropdownNode) {
        return;
      }

      if (!dropdownNode.contains(event.target)) {
        setIsTagFilterDropdownOpen(false);
      }
    };

    const handleDocumentKeyDown = (event) => {
      if (event.key === "Escape") {
        setIsTagFilterDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleDocumentMouseDown);
    document.addEventListener("keydown", handleDocumentKeyDown);

    return () => {
      document.removeEventListener("mousedown", handleDocumentMouseDown);
      document.removeEventListener("keydown", handleDocumentKeyDown);
    };
  }, [isTagFilterDropdownOpen]);

  useEffect(() => {
    if (!selectedNoDetectionImage) {
      return;
    }

    const stillExists = sortedNoDetectionImages.some(
      (image) => image.image_id === selectedNoDetectionImage.image_id
    );

    if (!stillExists) {
      setSelectedNoDetectionImage(null);
    }
  }, [sortedNoDetectionImages, selectedNoDetectionImage]);

  useEffect(() => {
    if (!masterDetectionsCheckboxRef.current) {
      return;
    }

    masterDetectionsCheckboxRef.current.indeterminate = areSomeVisibleDetectionsSelected;
  }, [areSomeVisibleDetectionsSelected]);

  useEffect(() => {
    if (!masterNoDetectionsCheckboxRef.current) {
      return;
    }

    masterNoDetectionsCheckboxRef.current.indeterminate = areSomeVisibleNoDetectionsSelected;
  }, [areSomeVisibleNoDetectionsSelected]);

  useEffect(() => {
    const availableImageIds = new Set(
      sortedNoDetectionImages.map((image) => String(image.image_id || "")).filter(Boolean)
    );

    setSelectedNoDetectionImageIds((previousSelectedIds) =>
      previousSelectedIds.filter((imageId) => availableImageIds.has(imageId))
    );

    if (expandedNoDetectionImageId && !availableImageIds.has(expandedNoDetectionImageId)) {
      setExpandedNoDetectionImageId(null);
    }
  }, [sortedNoDetectionImages, expandedNoDetectionImageId]);

  const handleResetHomeView = useCallback(() => {
    setCurrentLevel(0);
    setSelectedBBox(GEO_BOUNDS);
    setBBoxHistory([GEO_BOUNDS]);
    setGridCells(buildGridCells(GEO_BOUNDS, 0));
    setSelectedSegment(null);
    setFocusBounds(GEO_BOUNDS);
    setManualCoords({
      xMin: String(LON_MIN),
      yMin: String(LAT_MIN),
      xMax: String(LON_MAX),
      yMax: String(LAT_MAX),
    });
    setChosenMessage("Widok zresetowany do calej mapy.");
  }, []);

  const handleStepOutLevel = useCallback(() => {
    setBBoxHistory((prevHistory) => {
      if (prevHistory.length <= 1) {
        return prevHistory;
      }

      const nextHistory = prevHistory.slice(0, -1);
      const parentBounds = nextHistory[nextHistory.length - 1];

      // Ignore only the immediate zoom event from this programmatic fitBounds.
      suppressZoomOutRef.current = 1;

      setCurrentLevel(nextHistory.length - 1);
      setSelectedBBox(parentBounds);
      setFocusBounds(parentBounds);
      setSelectedSegment(null);
      setSelectedDetection(null);
      setHoveredDetectionId(null);

      const coords = boundsToCoords(parentBounds);
      setManualCoords({
        xMin: String(coords.xMin),
        yMin: String(coords.yMin),
        xMax: String(coords.xMax),
        yMax: String(coords.yMax),
      });

      return nextHistory;
    });
  }, []);

  const handleSelectSegment = (segment) => {
    setSelectedSegment(segment);
    setChosenMessage("");

    if (!isLevelLocked) {
      setCurrentLevel((prevLevel) => prevLevel + 1);
      setSelectedBBox(segment.bounds);
      setBBoxHistory((prevHistory) => [...prevHistory, segment.bounds]);
      setFocusBounds(segment.bounds);
    }

    const coords = boundsToCoords(segment.bounds);
    setManualCoords({
      xMin: String(coords.xMin),
      yMin: String(coords.yMin),
      xMax: String(coords.xMax),
      yMax: String(coords.yMax),
    });
  };

  const handleSelectDetection = useCallback((detection) => {
    const bboxBounds = detectionToBounds(detection);
    if (!bboxBounds) {
      return;
    }

    setSelectedDetection(detection);

    const mapInstance = mapRef.current;
    if (!mapInstance) {
      return;
    }

    mapInstance.fitBounds(bboxBounds, { padding: [20, 20], animate: true });
  }, []);

  const handleChooseArea = async () => {
    if (!selectedSegment) {
      setChosenMessage("Najpierw wybierz segment na mapie lub wpisz wspolrzedne.");
      setAnalysisStatus("error");
      return;
    }

    setIsLoadingDetections(true);
    setAnalysisStatus("loading");
    setAnalysisOverlayBounds(selectedSegment.bounds);

    const [[yMin, xMin], [yMax, xMax]] = selectedSegment.bounds;
    const analysisBbox = [xMin, yMin, xMax, yMax];

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          resolutionMode,
          numSamples,
          confidenceThreshold,
          bbox: analysisBbox,
        }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload && typeof errorPayload.detail === "string") {
            errorDetail = `${errorDetail}: ${errorPayload.detail}`;
          }
        } catch {
          // Ignore parse failures and keep default HTTP message.
        }
        throw new Error(errorDetail);
      }

      const runPayload = await response.json();
      const analysisId =
        runPayload && typeof runPayload === "object" && typeof runPayload.analysis_id === "string"
          ? runPayload.analysis_id
          : null;
      const runDetections =
        runPayload && typeof runPayload === "object" && Array.isArray(runPayload.detections)
          ? runPayload.detections
          : [];

      if (!analysisId) {
        throw new Error("Missing analysis_id in /analysis/run response");
      }

      const statusMap = await fetchDetectionStatuses();
      const detectionsWithStatus = applyStatusesToDetections(runDetections, statusMap);
      const visibleWithCurrentFilter = getDisplayDetectionsForStatus(
        detectionsWithStatus,
        statusFilter,
        {
          requireValidBounds: false,
          selectedTags: selectedTagFilters,
          tagMatchMode: tagFilterMode,
        }
      );

      if (detectionsWithStatus.length > 0 && visibleWithCurrentFilter.length === 0) {
        const fallbackStatus = ["to_verify", "confirmed", "rejected"].find(
          (candidateStatus) =>
            getDisplayDetectionsForStatus(detectionsWithStatus, candidateStatus, {
              requireValidBounds: false,
              selectedTags: selectedTagFilters,
              tagMatchMode: tagFilterMode,
            }).length > 0
        );

        if (fallbackStatus) {
          setStatusFilter(fallbackStatus);
        }
      }

      setCurrentAnalysisId(analysisId);
      setDetections(detectionsWithStatus);
      try {
        await fetchAnalysisImages();
      } catch (refreshError) {
        console.warn("Nie udalo sie odswiezyc listy zapisanych obrazow analizy:", refreshError);
      }
      setSelectedDetection(null);
      setAnalysisStatus("success");
      const statusSummary = formatDetectionStatusSummary(detectionsWithStatus);

      if (detectionsWithStatus.length === 0) {
        setChosenMessage(
          `Analiza ${analysisId} zakonczona. Brak detekcji dla wybranego obszaru. ${statusSummary}`
        );
        return;
      }

      showDetectedResultsInGallery(
        "Analiza",
        analysisId,
        detectionsWithStatus.length,
        statusSummary
      );
    } catch (error) {
      console.error("Blad podczas pobierania detekcji:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Nie udalo sie pobrac detekcji z backendu FastAPI.";
      if (isNoCoverageErrorMessage(errorMessage)) {
        setChosenMessage(
          "Brak pokrycia danych dla tej warstwy i obszaru. Zmien warstwe, zrodlo WMS lub zaznacz inny obszar."
        );
      } else {
        setChosenMessage(`Nie udalo sie pobrac detekcji. Szczegoly: ${errorMessage}`);
      }
      setAnalysisStatus("error");
      setDetections([]);
      setSelectedDetection(null);
    } finally {
      setIsLoadingDetections(false);
      setAnalysisOverlayBounds(null);
    }
  };

  const handleLocalAnalysis = async () => {
    setIsLoadingDetections(true);
    setAnalysisStatus("loading");
    setAnalysisOverlayBounds(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/local-run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          confidenceThreshold,
        }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload && typeof errorPayload.detail === "string") {
            errorDetail = `${errorDetail}: ${errorPayload.detail}`;
          }
        } catch {
          // Ignore parse failures and keep default HTTP message.
        }
        throw new Error(errorDetail);
      }

      const runPayload = await response.json();
      const analysisId =
        runPayload && typeof runPayload === "object" && typeof runPayload.analysis_id === "string"
          ? runPayload.analysis_id
          : null;
      const runDetections =
        runPayload && typeof runPayload === "object" && Array.isArray(runPayload.detections)
          ? runPayload.detections
          : [];

      if (!analysisId) {
        throw new Error("Missing analysis_id in /analysis/local-run response");
      }

      const statusMap = await fetchDetectionStatuses();
      const detectionsWithStatus = applyStatusesToDetections(runDetections, statusMap);
      const visibleWithCurrentFilter = getDisplayDetectionsForStatus(
        detectionsWithStatus,
        statusFilter,
        {
          requireValidBounds: false,
          selectedTags: selectedTagFilters,
          tagMatchMode: tagFilterMode,
        }
      );

      if (detectionsWithStatus.length > 0 && visibleWithCurrentFilter.length === 0) {
        const fallbackStatus = ["to_verify", "confirmed", "rejected"].find(
          (candidateStatus) =>
            getDisplayDetectionsForStatus(detectionsWithStatus, candidateStatus, {
              requireValidBounds: false,
              selectedTags: selectedTagFilters,
              tagMatchMode: tagFilterMode,
            }).length > 0
        );

        if (fallbackStatus) {
          setStatusFilter(fallbackStatus);
        }
      }

      setCurrentAnalysisId(analysisId);
      setDetections(detectionsWithStatus);
      setSelectedDetection(null);
      setAnalysisStatus("success");
      const statusSummary = formatDetectionStatusSummary(detectionsWithStatus);

      if (detectionsWithStatus.length === 0) {
        setChosenMessage(
          `Analiza lokalna ${analysisId} zakonczona. Brak detekcji w folderze validation. ${statusSummary}`
        );
        return;
      }

      showDetectedResultsInGallery(
        "Analiza lokalna",
        analysisId,
        detectionsWithStatus.length,
        statusSummary
      );
    } catch (error) {
      console.error("Blad podczas analizy lokalnej:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Nie udalo sie uruchomic analizy lokalnej z folderu validation.";
      setChosenMessage(`Nie udalo sie uruchomic analizy lokalnej. Szczegoly: ${errorMessage}`);
      setAnalysisStatus("error");
      setDetections([]);
      setSelectedDetection(null);
    } finally {
      setIsLoadingDetections(false);
    }
  };

  const handleGoToManual = () => {
    const xMin = Number(manualCoords.xMin);
    const yMin = Number(manualCoords.yMin);
    const xMax = Number(manualCoords.xMax);
    const yMax = Number(manualCoords.yMax);

    const hasInvalidValues =
      [xMin, yMin, xMax, yMax].some((value) => Number.isNaN(value)) ||
      xMin < LON_MIN ||
      yMin < LAT_MIN ||
      xMax > LON_MAX ||
      yMax > LAT_MAX ||
      xMax <= xMin ||
      yMax <= yMin;

    if (hasInvalidValues) {
      setChosenMessage(
        `Niepoprawne wspolrzedne. Zakres: lon ${LON_MIN} do ${LON_MAX}, lat ${LAT_MIN} do ${LAT_MAX}, oraz xMax>xMin i yMax>yMin.`
      );
      return;
    }

    const manualBounds = [
      [yMin, xMin],
      [yMax, xMax],
    ];

    setSelectedSegment({
      id: "manual",
      row: null,
      col: null,
      bounds: manualBounds,
    });
    setFocusBounds(manualBounds);
    setChosenMessage("Przejscie do recznie wskazanego obszaru.");
  };

  const handleStartEditComment = (detection) => {
    setSelectedDetection(detection);
    setExpandedDetectionId(detection.detection_id);
    setEditingDetectionId(detection.detection_id);
    setInputComment(detection.comment ?? "");
  };

  const handleSaveComment = async (detectionId) => {
    const targetDetectionId = editingDetectionId ?? detectionId;
    const currentComment = inputComment;

    if (!editingDetectionId && currentComment.trim().length === 0) {
      setChosenMessage("Wpisz komentarz przed zapisem.");
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/detections/${targetDetectionId}/comment`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ comment: currentComment }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();

      setDetections((prev) =>
        prev.map((detection) =>
          detection.detection_id === targetDetectionId
            ? { ...detection, comment: payload.comment }
            : detection
        )
      );

      setInputComment(payload.comment);
      setEditingDetectionId(targetDetectionId);
      setChosenMessage(editingDetectionId ? "Komentarz zaktualizowany." : "Komentarz zapisany.");
    } catch (error) {
      console.error("Blad podczas zapisu komentarza:", error);
      setChosenMessage("Nie udalo sie zapisac komentarza.");
    }
  };

  const handleDeleteComment = async (detectionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/detections/${detectionId}/comment`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ comment: "" }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setDetections((prev) =>
        prev.map((detection) =>
          detection.detection_id === detectionId
            ? { ...detection, comment: "" }
            : detection
        )
      );

      if (editingDetectionId === detectionId) {
        setInputComment("");
      }

      setChosenMessage("Komentarz usuniety.");
    } catch (error) {
      console.error("Blad podczas usuwania komentarza:", error);
      setChosenMessage("Nie udalo sie usunac komentarza.");
    }
  };

  const handleUpdateDetectionTags = async (detectionId, tags) => {
    const normalizedTags = normalizeDetectionTags(tags);

    const response = await fetch(`${API_BASE_URL}/detections/${detectionId}/tags`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ tags: normalizedTags }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    const updatedTags = normalizeDetectionTags(payload.tags);

    setDetections((prev) =>
      prev.map((detection) =>
        detection.detection_id === detectionId
          ? { ...detection, tags: updatedTags }
          : detection
      )
    );

    return updatedTags;
  };

  const handleAddTag = async (detectionId) => {
    const draftTag = String(tagDrafts[detectionId] ?? "").trim();
    if (!draftTag) {
      return;
    }

    const detection = detections.find((item) => item.detection_id === detectionId);
    const currentTags = normalizeDetectionTags(detection?.tags);
    const nextTags = normalizeDetectionTags([...currentTags, draftTag]);

    if (nextTags.length === currentTags.length) {
      setTagDrafts((prev) => ({ ...prev, [detectionId]: "" }));
      setChosenMessage("Tag juz istnieje.");
      return;
    }

    try {
      await handleUpdateDetectionTags(detectionId, nextTags);
      setTagDrafts((prev) => ({ ...prev, [detectionId]: "" }));
      setChosenMessage("Tag dodany.");
    } catch (error) {
      console.error("Blad podczas dodawania tagu:", error);
      setChosenMessage("Nie udalo sie dodac tagu.");
    }
  };

  const handleRemoveTag = async (detectionId, tagToRemove) => {
    const detection = detections.find((item) => item.detection_id === detectionId);
    const currentTags = normalizeDetectionTags(detection?.tags);
    const nextTags = currentTags.filter((tag) => tag !== tagToRemove);

    try {
      await handleUpdateDetectionTags(detectionId, nextTags);
      setChosenMessage("Tag usuniety.");
    } catch (error) {
      console.error("Blad podczas usuwania tagu:", error);
      setChosenMessage("Nie udalo sie usunac tagu.");
    }
  };

  const handleOpenDetectionPreviewModal = (detection) => {
    if (!detection) {
      return;
    }

    const previewImage = resolveAnalysisImageForDetection(detection, analysisImages);
    setDetectionPreviewImageMetrics({
      naturalWidth: 0,
      naturalHeight: 0,
      displayWidth: 0,
      displayHeight: 0,
    });
    setDetectionPreviewModal({
      isOpen: true,
      detection,
      image: previewImage,
      showBBoxPreview: true,
    });
  };

  const handleCloseDetectionPreviewModal = () => {
    setDetectionPreviewImageMetrics({
      naturalWidth: 0,
      naturalHeight: 0,
      displayWidth: 0,
      displayHeight: 0,
    });
    setDetectionPreviewModal((prevModal) => ({ ...prevModal, isOpen: false }));
  };

  const handleToggleDetectionPreviewBBox = () => {
    setDetectionPreviewModal((prevModal) => ({
      ...prevModal,
      showBBoxPreview: !prevModal.showBBoxPreview,
    }));
  };

  const handleToggleDetectionExpand = (detection) => {
    const detectionId = detection.detection_id;

    if (expandedDetectionId === detectionId) {
      setExpandedDetectionId(null);
      if (editingDetectionId === detectionId) {
        setEditingDetectionId(null);
        setInputComment("");
      }
      return;
    }

    setExpandedDetectionId(detectionId);
    setEditingDetectionId(detectionId);
    setInputComment(detection.comment ?? "");
  };

  const handleToggleDetectionSelection = (detectionId) => {
    setSelectedIds((prevSelectedIds) => {
      if (prevSelectedIds.includes(detectionId)) {
        return prevSelectedIds.filter((selectedId) => selectedId !== detectionId);
      }

      return [...prevSelectedIds, detectionId];
    });
  };

  const handleToggleSelectAllDetections = () => {
    if (selectableDetectionIds.length === 0) {
      return;
    }

    const selectableSet = new Set(selectableDetectionIds);
    setSelectedIds((previousSelectedIds) => {
      const normalizedPrevious = [
        ...new Set(
          previousSelectedIds
            .map((detectionId) => String(detectionId || "").trim())
            .filter(Boolean)
        ),
      ];

      if (areAllVisibleDetectionsSelected) {
        return normalizedPrevious.filter((detectionId) => !selectableSet.has(detectionId));
      }

      const nextSet = new Set(normalizedPrevious);
      for (const detectionId of selectableDetectionIds) {
        nextSet.add(detectionId);
      }

      return Array.from(nextSet);
    });
  };

  const handleApplyBulkTag = async () => {
    const normalizedTag = String(bulkTagDraft || "").trim();
    if (!normalizedTag) {
      setChosenMessage("Wpisz tag do dodania.");
      return;
    }

    const normalizedSelectedIds = [
      ...new Set(selectedIds.map((id) => String(id || "").trim()).filter(Boolean)),
    ];
    if (normalizedSelectedIds.length === 0) {
      setChosenMessage("Zaznacz co najmniej jedna detekcje do tagowania.");
      return;
    }

    setIsBulkTagging(true);
    try {
      const response = await fetch(`${API_BASE_URL}/detections/bulk/tags`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          detectionIds: normalizedSelectedIds,
          tag: normalizedTag,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      const appliedTag = String(payload?.tag ?? normalizedTag).trim() || normalizedTag;
      const updatedDetectionIds = Array.isArray(payload?.updated_detection_ids)
        ? payload.updated_detection_ids
            .map((id) => String(id || "").trim())
            .filter(Boolean)
        : [];
      const missingDetectionIds = Array.isArray(payload?.missing_detection_ids)
        ? payload.missing_detection_ids
            .map((id) => String(id || "").trim())
            .filter(Boolean)
        : [];
      const updatedDetectionIdsSet = new Set(updatedDetectionIds);

      if (updatedDetectionIdsSet.size > 0) {
        setDetections((prev) =>
          prev.map((detection) => {
            const detectionId = String(detection?.detection_id || "").trim();
            if (!updatedDetectionIdsSet.has(detectionId)) {
              return detection;
            }

            const nextTags = normalizeDetectionTags([
              ...normalizeDetectionTags(detection?.tags),
              appliedTag,
            ]);
            return { ...detection, tags: nextTags };
          })
        );
      }

      setBulkTagDraft("");
      if (updatedDetectionIds.length === 0) {
        setChosenMessage("Nie znaleziono wybranych detekcji do tagowania.");
        return;
      }

      const missingMessage =
        missingDetectionIds.length > 0
          ? ` Nie znaleziono ${missingDetectionIds.length} detekcji.`
          : "";
      setChosenMessage(
        `Dodano tag \"${appliedTag}\" do ${updatedDetectionIds.length} detekcji.${missingMessage}`
      );
    } catch (error) {
      console.error("Blad podczas masowego dodawania tagu:", error);
      setChosenMessage("Nie udalo sie dodac tagu do zaznaczonych detekcji.");
    } finally {
      setIsBulkTagging(false);
    }
  };

  const handleExportSelectedDetections = () => {
    const normalizedSelectedIds = [
      ...new Set(selectedIds.map((id) => String(id || "").trim()).filter(Boolean)),
    ];
    if (normalizedSelectedIds.length === 0) {
      setChosenMessage("Zaznacz co najmniej jedna detekcje do eksportu.");
      return;
    }

    const selectedIdsSet = new Set(normalizedSelectedIds);
    const selectedDetections = statusResolvedDetections.filter((detection) =>
      selectedIdsSet.has(String(detection?.detection_id || "").trim())
    );
    if (selectedDetections.length === 0) {
      setChosenMessage("Nie znaleziono wybranych detekcji do eksportu.");
      return;
    }

    const exportRows = selectedDetections.map((detection) => {
      const relatedImage = resolveAnalysisImageForDetection(detection, analysisImages);
      const bbox = detection?.bbox ?? {};
      return {
        id: String(detection?.detection_id || "").trim(),
        bbox: {
          x: Number(bbox?.x),
          y: Number(bbox?.y),
          width: Number(bbox?.width),
          height: Number(bbox?.height),
        },
        confidence: Number(detection?.confidence),
        status: String(detection?.status || "").trim(),
        tags: normalizeDetectionTags(detection?.tags),
        image_path: String(relatedImage?.path || detection?.path || "").trim(),
      };
    });

    const timestampToken = new Date().toISOString().replace(/[:.]/g, "-");
    const normalizedExportFormat = exportFormat === "csv" ? "csv" : "json";

    if (normalizedExportFormat === "csv") {
      const headerRow = ["id", "bbox", "confidence", "status", "tags", "image_path"];
      const csvRows = exportRows.map((row) => [
        row.id,
        JSON.stringify(row.bbox),
        row.confidence,
        row.status,
        JSON.stringify(row.tags),
        row.image_path,
      ]);

      const csvContent = [headerRow, ...csvRows]
        .map((cells) => cells.map((cell) => escapeCsvCell(cell)).join(","))
        .join("\n");

      triggerTextDownload(
        csvContent,
        `detections_export_${timestampToken}.csv`,
        "text/csv;charset=utf-8"
      );
      setChosenMessage(`Wyeksportowano ${exportRows.length} detekcji do CSV.`);
      return;
    }

    const jsonContent = JSON.stringify(exportRows, null, 2);
    triggerTextDownload(
      jsonContent,
      `detections_export_${timestampToken}.json`,
      "application/json;charset=utf-8"
    );
    setChosenMessage(`Wyeksportowano ${exportRows.length} detekcji do JSON.`);
  };

  const handleToggleTagFilter = (tag) => {
    setSelectedTagFilters((prevSelectedTags) => {
      if (prevSelectedTags.includes(tag)) {
        return prevSelectedTags.filter((selectedTag) => selectedTag !== tag);
      }

      return normalizeDetectionTags([...prevSelectedTags, tag]);
    });
  };

  const handleRemoveTagFilter = (tagToRemove) => {
    setSelectedTagFilters((prevSelectedTags) =>
      prevSelectedTags.filter((selectedTag) => selectedTag !== tagToRemove)
    );
  };

  const handleToggleNoDetectionSelection = (imageId) => {
    const normalizedImageId = String(imageId || "").trim();
    if (!normalizedImageId) {
      return;
    }

    setSelectedNoDetectionImageIds((prevSelectedIds) => {
      if (prevSelectedIds.includes(normalizedImageId)) {
        return prevSelectedIds.filter((selectedId) => selectedId !== normalizedImageId);
      }

      return [...prevSelectedIds, normalizedImageId];
    });
  };

  const handleToggleSelectAllNoDetections = () => {
    if (selectableNoDetectionImageIds.length === 0) {
      return;
    }

    const selectableSet = new Set(selectableNoDetectionImageIds);
    setSelectedNoDetectionImageIds((previousSelectedIds) => {
      const normalizedPrevious = [
        ...new Set(
          previousSelectedIds
            .map((imageId) => String(imageId || "").trim())
            .filter(Boolean)
        ),
      ];

      if (areAllVisibleNoDetectionsSelected) {
        return normalizedPrevious.filter((imageId) => !selectableSet.has(imageId));
      }

      const nextSet = new Set(normalizedPrevious);
      for (const imageId of selectableNoDetectionImageIds) {
        nextSet.add(imageId);
      }

      return Array.from(nextSet);
    });
  };

  const handleToggleNoDetectionExpand = (image) => {
    const imageId = String(image?.image_id || "").trim();
    if (!imageId) {
      return;
    }

    if (expandedNoDetectionImageId === imageId) {
      setExpandedNoDetectionImageId(null);
      return;
    }

    setExpandedNoDetectionImageId(imageId);
  };

  const handleApplyNoDetectionBulkTag = async () => {
    const normalizedTag = String(noDetectionBulkTagDraft || "").trim();
    if (!normalizedTag) {
      setChosenMessage("Wpisz tag do dodania.");
      return;
    }

    const normalizedSelectedIds = [
      ...new Set(
        selectedNoDetectionImageIds
          .map((imageId) => String(imageId || "").trim())
          .filter(Boolean)
      ),
    ];
    if (normalizedSelectedIds.length === 0) {
      setChosenMessage("Zaznacz co najmniej jeden obraz no_detections do tagowania.");
      return;
    }

    setIsNoDetectionBulkTagging(true);
    try {
      const response = await fetch(`${API_BASE_URL}/analysis-images/bulk/tags`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          imageIds: normalizedSelectedIds,
          tag: normalizedTag,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      const appliedTag = String(payload?.tag ?? normalizedTag).trim() || normalizedTag;
      const updatedImageIds = Array.isArray(payload?.updated_image_ids)
        ? payload.updated_image_ids.map((imageId) => String(imageId || "").trim()).filter(Boolean)
        : [];
      const missingImageIds = Array.isArray(payload?.missing_image_ids)
        ? payload.missing_image_ids.map((imageId) => String(imageId || "").trim()).filter(Boolean)
        : [];
      const updatedImageIdsSet = new Set(updatedImageIds);

      if (updatedImageIdsSet.size > 0) {
        setAnalysisImages((prev) =>
          prev.map((image) => {
            const imageId = String(image?.image_id || "").trim();
            if (!updatedImageIdsSet.has(imageId)) {
              return image;
            }

            const nextTags = normalizeDetectionTags([
              ...normalizeDetectionTags(image?.tags),
              appliedTag,
            ]);
            return { ...image, tags: nextTags };
          })
        );
      }

      setNoDetectionBulkTagDraft("");
      if (updatedImageIds.length === 0) {
        setChosenMessage("Nie znaleziono wybranych obrazow no_detections do tagowania.");
        return;
      }

      const missingMessage =
        missingImageIds.length > 0 ? ` Nie znaleziono ${missingImageIds.length} obrazow.` : "";
      setChosenMessage(
        `Dodano tag \"${appliedTag}\" do ${updatedImageIds.length} obrazow no_detections.${missingMessage}`
      );
    } catch (error) {
      console.error("Blad podczas masowego tagowania obrazow no_detections:", error);
      setChosenMessage("Nie udalo sie dodac tagu do zaznaczonych obrazow no_detections.");
    } finally {
      setIsNoDetectionBulkTagging(false);
    }
  };

  const handleRequestDeleteDetection = (detection) => {
    setDeleteModal({
      targetType: "detection",
      targetIds: [detection.detection_id],
      deleteImages: false,
      isDeleting: false,
      missingImageWarning: false,
    });
  };

  const handleRequestBulkDeleteDetections = () => {
    if (selectedIds.length === 0) {
      setChosenMessage("Zaznacz co najmniej jedna detekcje do usuniecia.");
      return;
    }

    setDeleteModal({
      targetType: "detection",
      targetIds: [...selectedIds],
      deleteImages: false,
      isDeleting: false,
      missingImageWarning: false,
    });
  };

  const handleRequestDeleteNoDetectionImage = (imageId) => {
    const normalizedImageId = String(imageId || "").trim();
    if (!normalizedImageId) {
      return;
    }

    setDeleteModal({
      targetType: "analysis_image",
      targetIds: [normalizedImageId],
      deleteImages: false,
      isDeleting: false,
      missingImageWarning: false,
    });
  };

  const handleRequestBulkDeleteNoDetectionImages = () => {
    const normalizedSelectedIds = selectedNoDetectionImageIds
      .map((imageId) => String(imageId || "").trim())
      .filter(Boolean);

    if (normalizedSelectedIds.length === 0) {
      setChosenMessage("Zaznacz co najmniej jeden obraz do usuniecia.");
      return;
    }

    setDeleteModal({
      targetType: "analysis_image",
      targetIds: [...new Set(normalizedSelectedIds)],
      deleteImages: false,
      isDeleting: false,
      missingImageWarning: false,
    });
  };

  const handleCancelDeleteDetection = () => {
    if (deleteModal.isDeleting) {
      return;
    }

    setDeleteModal({
      targetType: null,
      targetIds: [],
      deleteImages: false,
      isDeleting: false,
      missingImageWarning: false,
    });
  };

  const handleConfirmDeleteAction = async () => {
    if (deleteModal.targetIds.length === 0 || deleteModal.isDeleting) {
      return;
    }

    const targetType = deleteModal.targetType;
    if (targetType !== "detection" && targetType !== "analysis_image") {
      return;
    }

    const targetIds = [...deleteModal.targetIds];
    const isBulkDelete = targetIds.length > 1;
    const deleteImages = targetType === "detection" ? Boolean(deleteModal.deleteImages) : false;
    setDeleteModal((prev) => ({ ...prev, isDeleting: true }));

    let timeoutId = null;
    try {
      const abortController = new AbortController();
      const requestTimeoutMs = 30000;
      timeoutId = window.setTimeout(() => {
        abortController.abort();
      }, requestTimeoutMs);

      const endpoint =
        targetType === "detection"
          ? isBulkDelete
            ? `${API_BASE_URL}/detections/bulk`
            : `${API_BASE_URL}/detections/${encodeURIComponent(targetIds[0])}?deleteImages=${
                deleteImages ? "true" : "false"
              }`
          : isBulkDelete
            ? `${API_BASE_URL}/analysis-images/bulk`
            : `${API_BASE_URL}/analysis-images/${encodeURIComponent(targetIds[0])}?deleteFiles=${
                deleteImages ? "true" : "false"
              }`;

      const requestInit =
        isBulkDelete && targetType === "detection"
          ? {
              method: "DELETE",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                detectionIds: targetIds,
                deleteImages,
              }),
            }
          : isBulkDelete
            ? {
                method: "DELETE",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  imageIds: targetIds,
                  deleteFiles: deleteImages,
                }),
              }
            : {
                method: "DELETE",
              };

      const response = await fetch(endpoint, {
        ...requestInit,
        signal: abortController.signal,
      });
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload && typeof errorPayload.detail === "string") {
            errorDetail = `${errorDetail}: ${errorPayload.detail}`;
          }
        } catch {
          // Ignore parse errors and keep fallback detail.
        }

        throw new Error(errorDetail);
      }

      const payload = await response.json();
      if (targetType === "detection") {
        const deletedDetectionIds = isBulkDelete
          ? Array.isArray(payload?.deleted_detection_ids)
            ? payload.deleted_detection_ids.map((detectionId) => String(detectionId))
            : []
          : payload?.detection_deleted
            ? [targetIds[0]]
            : [];

        const missingDetectionIds = isBulkDelete
          ? Array.isArray(payload?.missing_detection_ids)
            ? payload.missing_detection_ids.map((detectionId) => String(detectionId))
            : []
          : [];

        const relatedImageMissing =
          deleteImages && typeof payload?.related_image_missing === "boolean"
            ? payload.related_image_missing
            : deleteImages && Boolean(
                !isBulkDelete &&
                  (!payload?.deleted_image_id || String(payload.deleted_image_id).length === 0)
              );
        const relatedImageInUse =
          deleteImages && typeof payload?.related_image_in_use === "boolean"
            ? payload.related_image_in_use
            : false;
        const relatedImageMissingCount =
          deleteImages && isBulkDelete
            ? Number(payload?.related_image_missing_count || 0)
            : relatedImageMissing
              ? 1
              : 0;
        const relatedImageInUseCount =
          deleteImages && isBulkDelete
            ? Number(payload?.related_image_in_use_count || 0)
            : relatedImageInUse
              ? 1
              : 0;

        const deletedIdsSet = new Set(deletedDetectionIds);

        if (deletedIdsSet.size > 0) {
          setDetections((prev) =>
            prev.filter((detection) => !deletedIdsSet.has(detection.detection_id))
          );
          setStoredStatuses((prev) => {
            const next = { ...prev };
            for (const detectionId of deletedIdsSet) {
              delete next[detectionId];
            }
            return next;
          });
          setSelectedIds((prevSelectedIds) =>
            prevSelectedIds.filter((detectionId) => !deletedIdsSet.has(detectionId))
          );
          setTagDrafts((prevTagDrafts) => {
            const nextDrafts = { ...prevTagDrafts };
            for (const detectionId of deletedIdsSet) {
              delete nextDrafts[detectionId];
            }
            return nextDrafts;
          });

          if (selectedDetection && deletedIdsSet.has(selectedDetection.detection_id)) {
            setSelectedDetection(null);
          }

          if (expandedDetectionId && deletedIdsSet.has(expandedDetectionId)) {
            setExpandedDetectionId(null);
          }

          if (editingDetectionId && deletedIdsSet.has(editingDetectionId)) {
            setEditingDetectionId(null);
            setInputComment("");
          }

          if (hoveredDetectionId) {
            const hoveredDetectionIdParts = hoveredDetectionId.split("|");
            const hoveredDetectionRawId = hoveredDetectionIdParts[1] ?? "";
            if (deletedIdsSet.has(hoveredDetectionRawId)) {
              setHoveredDetectionId(null);
            }
          }
        }

        try {
          await fetchAnalysisImages();
        } catch (refreshError) {
          console.warn("Nie udalo sie odswiezyc obrazow po usunieciu detekcji:", refreshError);
        }

        setDeleteModal({
          targetType: null,
          targetIds: [],
          deleteImages: false,
          isDeleting: false,
          missingImageWarning: false,
        });

        if (isBulkDelete) {
          const deletedCount = deletedDetectionIds.length;
          const missingCount = missingDetectionIds.length;
          const missingMessage =
            missingCount > 0 ? ` Nie znaleziono ${missingCount} detekcji.` : "";
          if (!deleteImages) {
            setChosenMessage(
              `Usunieto ${deletedCount} zaznaczonych detekcji. Obrazy pozostawiono.${missingMessage}`
            );
          } else {
            const imageMissingMessage =
              relatedImageMissingCount > 0
                ? ` Brak pliku obrazu dla ${relatedImageMissingCount} detekcji.`
                : "";
            const imageInUseMessage =
              relatedImageInUseCount > 0
                ? ` Obraz uzywany przez inne detekcje dla ${relatedImageInUseCount} pozycji - plik pozostawiono.`
                : "";
            setChosenMessage(
              `Usunieto ${deletedCount} zaznaczonych detekcji.${missingMessage}${imageMissingMessage}${imageInUseMessage}`
            );
          }
        } else {
          if (!deleteImages) {
            setChosenMessage("Detekcja usunieta. Powiazany obraz pozostawiono.");
          } else if (relatedImageInUse) {
            setChosenMessage(
              "Detekcja usunieta. Obraz jest uzywany przez inne detekcje, wiec nie zostal usuniety."
            );
          } else {
            setChosenMessage(
              relatedImageMissing
                ? "Detekcja usunieta. Powiazany obraz nie byl dostepny."
                : "Detekcja i powiazany obraz zostaly usuniete."
            );
          }
        }
      } else {
        const deletedImageIds = isBulkDelete
          ? Array.isArray(payload?.deleted_image_ids)
            ? payload.deleted_image_ids.map((imageId) => String(imageId))
            : []
          : payload?.image_deleted
            ? [targetIds[0]]
            : [];

        const missingImageIds = isBulkDelete
          ? Array.isArray(payload?.missing_image_ids)
            ? payload.missing_image_ids.map((imageId) => String(imageId))
            : []
          : [];

        const deletedImageIdsSet = new Set(deletedImageIds);

        if (deletedImageIdsSet.size > 0) {
          setAnalysisImages((prev) =>
            prev.filter((image) => !deletedImageIdsSet.has(String(image.image_id || "")))
          );
          setSelectedNoDetectionImageIds((prevSelectedIds) =>
            prevSelectedIds.filter((imageId) => !deletedImageIdsSet.has(imageId))
          );

          if (
            selectedNoDetectionImage &&
            deletedImageIdsSet.has(String(selectedNoDetectionImage.image_id || ""))
          ) {
            setSelectedNoDetectionImage(null);
          }

          if (expandedNoDetectionImageId && deletedImageIdsSet.has(expandedNoDetectionImageId)) {
            setExpandedNoDetectionImageId(null);
          }
        }

        setDeleteModal({
          targetType: null,
          targetIds: [],
          deleteImages: false,
          isDeleting: false,
          missingImageWarning: false,
        });

        if (isBulkDelete) {
          const deletedCount = deletedImageIds.length;
          const missingCount = missingImageIds.length;
          const missingMessage = missingCount > 0 ? ` Nie znaleziono ${missingCount} obrazow.` : "";
          if (deleteImages) {
            setChosenMessage(`Usunieto ${deletedCount} zaznaczonych obrazow.${missingMessage}`);
          } else {
            setChosenMessage(
              `Usunieto ${deletedCount} zaznaczonych wpisow obrazow. Pliki pozostawiono.${missingMessage}`
            );
          }
        } else {
          setChosenMessage(
            deleteImages
              ? "Obraz zostal usuniety."
              : "Wpis obrazu zostal usuniety. Plik pozostawiono."
          );
        }
      }
    } catch (error) {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
      console.error("Blad podczas usuwania:", error);
      setDeleteModal((prev) => ({ ...prev, isDeleting: false }));
      const timeoutHint =
        error instanceof DOMException && error.name === "AbortError"
          ? " Zadanie przekroczylo limit czasu."
          : "";
      setChosenMessage(
        `Nie udalo sie usunac elementu.${timeoutHint} ${error instanceof Error ? error.message : ""}`.trim()
      );
    }
  };

  const handleOpenAnalysisImage = useCallback((image) => {
    setSelectedNoDetectionImage(image);
    setViewMode("gallery");
  }, []);

  return (
    <div className="container-fluid py-3 app-shell">
      <div className="row g-3 app-main-row">
        <div className="col-lg-6 col-xl-8">
          <div className={viewMode === "map" ? "" : "d-none"}>
            <div className="map-shell border rounded shadow-sm">
              <MapContainer
                crs={L.CRS.EPSG4326}
                bounds={GEO_BOUNDS}
                maxBounds={GEO_BOUNDS}
                maxBoundsViscosity={1.0}
                minZoom={0}
                maxZoom={18}
                zoomSnap={0.1}
                zoomDelta={0.5}
                whenCreated={(mapInstance) => {
                  mapRef.current = mapInstance;
                }}
              >
                <WMSTileLayer
                  url="https://planetarymaps.usgs.gov/cgi-bin/mapserv"
                  map="/maps/earth/moon_simp_cyl.map"
                  service="WMS"
                  version="1.1.1"
                  crs={L.CRS.EPSG4326}
                  layers="KaguyaTC_Ortho"
                  format="image/png"
                  transparent={false}
                  noWrap
                />
                <ZoomOutLevelControl
                  currentLevel={currentLevel}
                  onStepOut={handleStepOutLevel}
                  suppressZoomOutRef={suppressZoomOutRef}
                />
                <FitBoundsOnChange bounds={focusBounds} />
                <HomeControl onHomeClick={handleResetHomeView} />

                {gridCells.map((segment) => {
                  const isHovered = hoveredSegmentId === segment.id;
                  const isSelected = selectedSegment?.id === segment.id;

                  let color = "#0d6efd";
                  if (isSelected) {
                    color = "#dc3545";
                  } else if (isHovered) {
                    color = "#fd7e14";
                  }

                  return (
                    <Rectangle
                      key={segment.id}
                      bounds={segment.bounds}
                      pathOptions={{
                        color,
                        weight: isSelected ? 2 : 1,
                        opacity: isSelected ? 0.8 : isHovered ? 0.55 : 0.32,
                        fillColor: color,
                        fillOpacity: isSelected ? 0.14 : isHovered ? 0.08 : 0.04,
                      }}
                      eventHandlers={{
                        mouseover: () => setHoveredSegmentId(segment.id),
                        mouseout: () => setHoveredSegmentId(null),
                        click: () => handleSelectSegment(segment),
                      }}
                    />
                  );
                })}

                {isAnalysisLoading && analysisOverlayBounds && (
                  <Rectangle
                    bounds={analysisOverlayBounds}
                    pathOptions={{
                      color: "#0d6efd",
                      weight: 1,
                      opacity: 0.75,
                      fillColor: "#0d6efd",
                      fillOpacity: 0.24,
                      dashArray: "6 4",
                      interactive: false,
                    }}
                  />
                )}

                {selectedDetection && (
                  <Rectangle
                    key={`overlay-${getDetectionUniqueId(selectedDetection)}`}
                    bounds={detectionToBounds(selectedDetection)}
                    pathOptions={{
                      color: "#fd7e14",
                      weight: 0,
                      fillColor: "#fd7e14",
                      fillOpacity: 0.28,
                      interactive: false,
                    }}
                  />
                )}

                {showBboxes && mapDetections.map((detection, detectionIndex) => {
                  const detectionUniqueId = getDetectionUniqueId(detection);
                  const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                  const isSelected = isSameDetection(selectedDetection, detection);
                  const isHovered = hoveredDetectionId === detectionUniqueId;
                  const statusColor = getStatusColor(detection.status);

                  return (
                    <Rectangle
                      key={detectionRenderKey}
                      bounds={detectionToBounds(detection)}
                      pathOptions={{
                        color: statusColor,
                        weight: isSelected ? 5 : isHovered ? 4 : 3,
                        opacity: isSelected ? 0.95 : isHovered ? 0.9 : 0.85,
                        fillColor: statusColor,
                        fillOpacity: isSelected ? 0.3 : isHovered ? 0.2 : 0.14,
                        dashArray: isSelected || isHovered ? null : "5 4",
                      }}
                      eventHandlers={{
                        mouseover: () => setHoveredDetectionId(detectionUniqueId),
                        mouseout: () => setHoveredDetectionId(null),
                        click: () => handleSelectDetection(detection),
                      }}
                    />
                  );
                })}
              </MapContainer>
            </div>
          </div>

          {viewMode === "gallery" && (
            <div className="gallery-shell border rounded shadow-sm p-3">
              <h5 className="mb-1">Przegladarka zdjec</h5>
              {selectedNoDetectionImage ? (
                <div className="mb-3">
                  <div className="d-flex justify-content-between align-items-center gap-2 flex-wrap mb-2">
                    <div className="small text-muted">
                      Podglad wybranego zapisanego zdjecia analizy.
                    </div>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => setSelectedNoDetectionImage(null)}
                    >
                      Zamknij podglad
                    </button>
                  </div>
                  <div className="gallery-focus-card border rounded p-2 bg-white">
                    <img
                      src={getAnalysisImageUrl(selectedNoDetectionImage.image_id)}
                      alt={`Zapis analizy ${selectedNoDetectionImage.image_id}`}
                      className="gallery-focus-image"
                    />
                    <div className="small text-muted mt-2">
                      <div><strong>ID:</strong> {selectedNoDetectionImage.image_id}</div>
                      <div><strong>Status:</strong> {selectedNoDetectionImage.status || NO_DETECTIONS_FILTER}</div>
                      <div>
                        <strong>Lat/Lon:</strong>{" "}
                        {typeof selectedNoDetectionImage.lat === "number"
                          ? selectedNoDetectionImage.lat.toFixed(6)
                          : "-"}
                        {" / "}
                        {typeof selectedNoDetectionImage.lon === "number"
                          ? selectedNoDetectionImage.lon.toFixed(6)
                          : "-"}
                      </div>
                      <div><strong>Rozdzielczosc:</strong> {selectedNoDetectionImage.resolution || "-"}</div>
                      {selectedNoDetectionImage.timestamp && (
                        <div><strong>Czas:</strong> {selectedNoDetectionImage.timestamp}</div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="small text-muted mb-3">
                  {isNoDetectionsFilterSelected
                    ? "Widok kart oparty o zapisane obrazy analizy."
                    : "Widok kart oparty o aktualnie widoczne detekcje (filtry statusu i tagow sa zachowane)."}
                </div>
              )}

              {selectedNoDetectionImage ? null : isNoDetectionsFilterSelected ? (
                sortedNoDetectionImages.length === 0 ? (
                  <div className="small text-muted">Brak zapisanych obrazow z wynikiem no_detections.</div>
                ) : (
                  <div className="row g-3">
                    {sortedNoDetectionImages.map((image, imageIndex) => {
                      const itemKey = `${image.image_id || "no-id"}|${image.timestamp || "no-ts"}|${imageIndex}`;
                      const isNoDetectionsStatus =
                        (typeof image.status === "string" ? image.status : NO_DETECTIONS_FILTER) ===
                        NO_DETECTIONS_FILTER;
                      const isSelected = selectedNoDetectionImageIds.includes(image.image_id);
                      const isHovered = hoveredDetectionId === image.image_id;

                      return (
                        <div className="col-12 col-sm-6 col-lg-4 col-xl-3" key={`gallery-image-${itemKey}`}
                          onMouseEnter={() => setHoveredDetectionId(image.image_id)}
                          onMouseLeave={() => setHoveredDetectionId(null)}
                        >
                          <div
                            className={`gallery-card card h-100 shadow-sm border-0 position-relative${isSelected ? " gallery-card-selected" : ""}`}
                            tabIndex={0}
                            onClick={() => handleToggleNoDetectionSelection(image.image_id)}
                            onKeyDown={e => {
                              if (e.key === "Enter" || e.key === " ") handleToggleNoDetectionSelection(image.image_id);
                            }}
                            style={{ cursor: "pointer" }}
                          >
                            <div className="gallery-img-wrap position-relative">
                              <img
                                src={getAnalysisImageUrl(image.image_id)}
                                alt={`Zapis analizy ${image.image_id}`}
                                loading="lazy"
                                className="gallery-preview card-img-top"
                              />
                              {isSelected && (
                                <div className="gallery-selected-overlay">
                                  <span className="gallery-checkmark">✔</span>
                                </div>
                              )}
                              {(isHovered || isSelected) && (
                                <div className="gallery-hover-icons">
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-secondary gallery-icon-btn"
                                    title="Podglad"
                                    onClick={e => {
                                      e.stopPropagation();
                                      handleOpenAnalysisImage(image);
                                    }}
                                  >
                                    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                      <path fill="currentColor" d="M16 8s-3-5.5-8-5.5S0 8 0 8s3 5.5 8 5.5S16 8 16 8zM1.173 8a13.133 13.133 0 0 1 1.66-2.043C4.12 4.668 5.88 3.5 8 3.5s3.879 1.168 5.168 2.457A13.133 13.133 0 0 1 14.828 8c-.058.087-.122.183-.195.288-.335.48-.83 1.12-1.465 1.755C11.879 11.332 10.119 12.5 8 12.5s-3.879-1.168-5.168-2.457A13.134 13.134 0 0 1 1.172 8z" />
                                      <path fill="currentColor" d="M8 5.5A2.5 2.5 0 1 0 8 10.5 2.5 2.5 0 0 0 8 5.5zM4.5 8a3.5 3.5 0 1 1 7 0 3.5 3.5 0 0 1-7 0z" />
                                    </svg>
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-danger gallery-icon-btn"
                                    title="Usuń"
                                    onClick={e => {
                                      e.stopPropagation();
                                      handleRequestDeleteNoDetectionImage(image.image_id);
                                    }}
                                  >
                                    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                      <path fill="currentColor" d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6zM1 3.5A.5.5 0 0 1 1.5 3H4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1h2.5a.5.5 0 0 1 0 1h-.538l-.853 10.66A1 1 0 0 1 12.112 15H3.888a1 1 0 0 1-.997-.84L2.038 4.5H1.5a.5.5 0 0 1-.5-.5zM5 2v1h6V2H5z" />
                                    </svg>
                                  </button>
                                </div>
                              )}
                            </div>
                            <div className="card-body py-2">
                              <div className="fw-semibold small">{image.image_id || "analysis_image"}</div>
                              <div className="small mt-1">
                                <span
                                  className={`badge ${isNoDetectionsStatus ? "text-bg-secondary" : "text-bg-primary"}`}
                                >
                                  {image.status || NO_DETECTIONS_FILTER}
                                </span>
                              </div>
                              <div className="small text-muted">
                                rozdzielczosc: {image.resolution || "-"}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )
              ) : detections.length === 0 ? (
                <div className="small text-muted">Brak detekcji. Kliknij "Uruchom analize".</div>
              ) : filteredDetections.length === 0 ? (
                <div className="small text-muted">
                  Brak detekcji dla statusu: {statusFilter}
                  {selectedTagFilters.length > 0
                    ? ` (tagi: ${selectedTagFilters.join(", ")} - ${tagFilterMode.toUpperCase()})`
                    : ""}
                  .
                </div>
              ) : (
                <div className="row g-3">
                  {filteredDetections.map((detection, detectionIndex) => {
                    const detectionUniqueId = getDetectionUniqueId(detection);
                    const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                    const statusBadgeClass = getStatusBadgeClass(detection.status);
                    const previewImage = resolveAnalysisImageForDetection(detection, analysisImages);
                    const previewUrl = previewImage ? getAnalysisImageUrl(previewImage.image_id) : getDetectionPreviewUrl(detection);
                    const isSelected = selectedIds.includes(detection.detection_id);
                    const isHovered = hoveredDetectionId === detectionUniqueId;

                    return (
                      <div
                        className="col-sm-6 col-xl-4"
                        key={`gallery-${detectionRenderKey}`}
                      >
                        <div
                          className={`gallery-card card h-100 shadow-sm border-0 position-relative${isSelected ? " gallery-card-selected" : ""}`}
                          tabIndex={0}
                          onClick={() => handleToggleDetectionSelection(detection.detection_id)}
                          onKeyDown={e => {
                            if (e.key === "Enter" || e.key === " ") handleToggleDetectionSelection(detection.detection_id);
                          }}
                          onMouseEnter={() => setHoveredDetectionId(detectionUniqueId)}
                          onMouseLeave={() => setHoveredDetectionId(null)}
                          style={{ cursor: "pointer" }}
                        >
                          <div className="gallery-img-wrap position-relative">
                            {previewUrl ? (
                              <img
                                src={previewUrl}
                                alt={`Podglad detekcji ${detection.detection_id}`}
                                loading="lazy"
                                className="gallery-preview card-img-top"
                              />
                            ) : (
                              <div className="gallery-preview-fallback card-img-top">Brak podgladu</div>
                            )}
                            {isSelected && (
                              <div className="gallery-selected-overlay">
                                <span className="gallery-checkmark">✔</span>
                              </div>
                            )}
                            {(isHovered || isSelected) && (
                              <div className="gallery-hover-icons">
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-secondary gallery-icon-btn"
                                  title="Podglad"
                                  onClick={e => {
                                    e.stopPropagation();
                                    handleOpenDetectionPreviewModal(detection);
                                  }}
                                >
                                  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                    <path fill="currentColor" d="M16 8s-3-5.5-8-5.5S0 8 0 8s3 5.5 8 5.5S16 8 16 8zM1.173 8a13.133 13.133 0 0 1 1.66-2.043C4.12 4.668 5.88 3.5 8 3.5s3.879 1.168 5.168 2.457A13.133 13.133 0 0 1 14.828 8c-.058.087-.122.183-.195.288-.335.48-.83 1.12-1.465 1.755C11.879 11.332 10.119 12.5 8 12.5s-3.879-1.168-5.168-2.457A13.134 13.134 0 0 1 1.172 8z" />
                                    <path fill="currentColor" d="M8 5.5A2.5 2.5 0 1 0 8 10.5 2.5 2.5 0 0 0 8 5.5zM4.5 8a3.5 3.5 0 1 1 7 0 3.5 3.5 0 0 1-7 0z" />
                                  </svg>
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-danger gallery-icon-btn"
                                  title="Usuń"
                                  onClick={e => {
                                    e.stopPropagation();
                                    handleRequestDeleteDetection(detection);
                                  }}
                                >
                                  <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                    <path fill="currentColor" d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6zM1 3.5A.5.5 0 0 1 1.5 3H4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1h2.5a.5.5 0 0 1 0 1h-.538l-.853 10.66A1 1 0 0 1 12.112 15H3.888a1 1 0 0 1-.997-.84L2.038 4.5H1.5a.5.5 0 0 1-.5-.5zM5 2v1h6V2H5z" />
                                  </svg>
                                </button>
                              </div>
                            )}
                          </div>
                          <div className="card-body py-2">
                            <div className="fw-semibold small">{detection.detection_id}</div>
                            <div className="small mt-1">
                              <span className={`badge ${statusBadgeClass}`}>{detection.status}</span>
                            </div>
                            <div className="small text-muted">
                              confidence: {Number(detection.confidence).toFixed(2)}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="col-lg-6 col-xl-4 app-sidebar-column">
          <div className="card shadow-sm sidebar-card">
            <div className="card-body sidebar-card-body">
              <h5 className="card-title">Panel obszaru</h5>

              <div className="small text-muted mb-2">Widok aplikacji</div>
              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Tryb widoku aplikacji">
                <button
                  type="button"
                  className={`btn ${viewMode === "map" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setViewMode("map")}
                >
                  Mapa
                </button>
                <button
                  type="button"
                  className={`btn ${viewMode === "gallery" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setViewMode("gallery")}
                >
                  Przegladarka zdjec
                </button>
              </div>

              <div className="small text-muted mb-2">Wybrany segment</div>
              {selectedSegment ? (
                <div className="mb-3">
                  <div><strong>ID:</strong> {selectedSegment.id}</div>
                  {selectedCoords && (
                    <div className="mt-2">
                      <div>xMin: {selectedCoords.xMin}</div>
                      <div>yMin: {selectedCoords.yMin}</div>
                      <div>xMax: {selectedCoords.xMax}</div>
                      <div>yMax: {selectedCoords.yMax}</div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mb-3 text-muted">Brak wybranego segmentu.</div>
              )}

              <button
                className="btn btn-primary w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                onClick={handleChooseArea}
                disabled={isLoadingDetections}
              >
                {isLoadingDetections && (
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                )}
                <span>{isLoadingDetections ? "Analizowanie..." : "Uruchom analize"}</span>
              </button>

              <button
                className="btn btn-outline-primary w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                onClick={handleLocalAnalysis}
                disabled={isLoadingDetections}
              >
                {isLoadingDetections && (
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                )}
                <span>{isLoadingDetections ? "Analizowanie..." : "Analiza lokalna"}</span>
              </button>

              <button className="btn btn-outline-secondary w-100 mb-2" onClick={handleResetHomeView}>
                Reset widoku
              </button>

              <div className="form-check form-switch mb-2">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="toggle-lock-level"
                  checked={isLevelLocked}
                  onChange={(event) => setIsLevelLocked(event.target.checked)}
                />
                <label className="form-check-label" htmlFor="toggle-lock-level">
                  Lock level
                </label>
              </div>

              <div className="small text-muted mb-2">
                {isLevelLocked ? "Klik tylko zaznacza segment." : "Klik schodzi poziom nizej."}
              </div>

              <div className="small text-muted mb-3">Poziom siatki: {currentLevel}</div>

              {analysisStatus && (
                <div className="small mb-3">
                  Status analizy:{" "}
                  <span
                    className={`badge ${
                      analysisStatus === "loading"
                        ? "text-bg-info"
                        : analysisStatus === "success"
                          ? "text-bg-success"
                          : "text-bg-danger"
                    }`}
                  >
                    {analysisStatus}
                  </span>
                </div>
              )}

              {currentAnalysisId && (
                <div className="small text-muted mb-3">analysis_id: {currentAnalysisId}</div>
              )}

              <div className="small text-muted mb-2">Rozdzielczosc analizy</div>
              <div className="btn-group btn-group-sm w-100 mb-2" role="group" aria-label="Tryb rozdzielczosci analizy">
                <button
                  type="button"
                  className={`btn ${resolutionMode === "preview" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("preview")}
                >
                  preview
                </button>
                <button
                  type="button"
                  className={`btn ${resolutionMode === "detail" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("detail")}
                >
                  detail
                </button>
                <button
                  type="button"
                  className={`btn ${resolutionMode === "ultra" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("ultra")}
                >
                  ultra
                </button>
              </div>
              <div className="small text-muted mb-3">
                {RESOLUTION_DESCRIPTION_MAP[resolutionMode]} (ok. {RESOLUTION_MPP_MAP[resolutionMode].toFixed(2)} mpp)
              </div>

              <div className="small text-muted mb-3">Zrodlo i warstwa WMS: USGS / KaguyaTC_Ortho</div>

              <div className="small text-muted mb-2">Liczba probek</div>
              <div className="btn-group btn-group-sm w-100 mb-2" role="group" aria-label="Liczba probek analizy">
                <button
                  type="button"
                  className={`btn ${numSamples === 1 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(1)}
                >
                  1
                </button>
                <button
                  type="button"
                  className={`btn ${numSamples === 5 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(5)}
                >
                  5
                </button>
                <button
                  type="button"
                  className={`btn ${numSamples === 10 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(10)}
                >
                  10
                </button>
              </div>
              <div className="small text-muted mb-3">Wiecej probek = dluzsza analiza</div>

              <div className="small text-muted mb-2">Confidence threshold ({confidenceThreshold.toFixed(2)})</div>
              <input
                className="form-range mb-3"
                type="range"
                min="0.01"
                max="1.0"
                step="0.01"
                value={confidenceThreshold}
                onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
              />

              <div className="small text-muted mb-2">Przejdz do wspolrzednych</div>
              <div className="row g-2">
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.xMin}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, xMin: event.target.value }))
                    }
                    placeholder="xMin"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.yMin}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, yMin: event.target.value }))
                    }
                    placeholder="yMin"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.xMax}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, xMax: event.target.value }))
                    }
                    placeholder="xMax"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.yMax}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, yMax: event.target.value }))
                    }
                    placeholder="yMax"
                  />
                </div>
              </div>

              <button className="btn btn-outline-secondary w-100 mt-3" onClick={handleGoToManual}>
                Przejdz
              </button>

              <div className="form-check form-switch mt-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="toggle-bboxes"
                  checked={showBboxes}
                  onChange={(event) => setShowBboxes(event.target.checked)}
                />
                <label className="form-check-label" htmlFor="toggle-bboxes">
                  Pokaz bounding boxy
                </label>
              </div>

              {chosenMessage && (
                <div
                  className={`alert ${isNoCoverageChosenMessage ? "alert-warning" : "alert-info"} py-2 mt-3 mb-0`}
                >
                  {chosenMessage}
                </div>
              )}

              <hr className="my-4" />
              <h6 className="mb-3">Detekcje ({detectionSectionCount})</h6>

              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Filtr statusu detekcji">
                <button
                  type="button"
                  className={`btn ${statusFilter === "confirmed" ? "btn-success" : "btn-outline-success"}`}
                  onClick={() => setStatusFilter("confirmed")}
                >
                  confirmed
                </button>
                <button
                  type="button"
                  className={`btn ${statusFilter === "to_verify" ? "btn-warning" : "btn-outline-warning"}`}
                  onClick={() => setStatusFilter("to_verify")}
                >
                  to_verify
                </button>
                <button
                  type="button"
                  className={`btn ${statusFilter === "rejected" ? "btn-danger" : "btn-outline-danger"}`}
                  onClick={() => setStatusFilter("rejected")}
                >
                  rejected
                </button>
                <button
                  type="button"
                  className={`btn ${statusFilter === NO_DETECTIONS_FILTER ? "btn-secondary" : "btn-outline-secondary"}`}
                  onClick={() => setStatusFilter(NO_DETECTIONS_FILTER)}
                >
                  no_detections
                </button>
              </div>

              <div className="mb-3">
                <div className="small text-muted mb-1">Sortowanie listy detekcji</div>
                {availableQuickSortFields.length === 0 ? (
                  <div className="small text-muted">Brak danych do sortowania.</div>
                ) : (
                  <div className="detection-sort-quick-list">
                    {availableQuickSortFields.map((field) => {
                      const isActive = effectiveSortBy === field.key;
                      const directionIcon = effectiveSortOrder === "asc" ? "▲" : "▼";

                      return (
                        <button
                          type="button"
                          key={`quick-sort-${field.key}`}
                          className={`btn btn-sm ${
                            isActive ? "btn-secondary" : "btn-outline-secondary"
                          } detection-sort-quick-btn`}
                          onClick={() => handleSortFieldClick(field.key)}
                          aria-pressed={isActive}
                          title={isActive ? `Kierunek: ${effectiveSortOrder}` : "Ustaw sortowanie"}
                        >
                          <span>{field.label}</span>
                          {isActive && <span className="detection-sort-quick-icon">{directionIcon}</span>}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {!isNoDetectionsFilterSelected && (
                <div className="mb-3">
                  <div className="d-flex justify-content-between align-items-center mb-2">
                    <div className="small text-muted">Filtr tagow</div>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary py-0"
                      onClick={() => {
                        setSelectedTagFilters([]);
                        setIsTagFilterDropdownOpen(false);
                      }}
                      disabled={selectedTagFilters.length === 0}
                    >
                      Wyczysc
                    </button>
                  </div>

                  {availableDetectionTags.length === 0 ? (
                    <div className="small text-muted">Brak dostepnych tagow dla wybranego statusu.</div>
                  ) : (
                    <>
                      <div className="dropdown tag-filter-dropdown mb-2" ref={tagFilterDropdownRef}>
                        <button
                          type="button"
                          className="btn btn-sm btn-outline-secondary w-100 d-flex justify-content-between align-items-center"
                          onClick={() => setIsTagFilterDropdownOpen((prevState) => !prevState)}
                          aria-expanded={isTagFilterDropdownOpen}
                          aria-haspopup="listbox"
                        >
                          <span>
                            {selectedTagFilters.length > 0
                              ? `Wybrane tagi (${selectedTagFilters.length})`
                              : "Wybierz tagi"}
                          </span>
                          <span className="small text-muted">
                            {isTagFilterDropdownOpen ? "Zamknij" : "Otworz"}
                          </span>
                        </button>

                        {isTagFilterDropdownOpen && (
                          <div className="dropdown-menu d-block w-100 mt-1 p-2 tag-filter-dropdown-menu">
                            {availableDetectionTags.map((tag) => {
                              const isChecked = selectedTagFilters.includes(tag);

                              return (
                                <label key={`tag-filter-${tag}`} className="form-check mb-1 tag-filter-option">
                                  <input
                                    className="form-check-input"
                                    type="checkbox"
                                    checked={isChecked}
                                    onChange={() => handleToggleTagFilter(tag)}
                                  />
                                  <span className="form-check-label small">{tag}</span>
                                </label>
                              );
                            })}
                          </div>
                        )}
                      </div>

                      {selectedTagFilters.length > 0 ? (
                        <div className="d-flex flex-wrap gap-1 mb-2">
                          {selectedTagFilters.map((tag) => (
                            <span
                              key={`active-tag-filter-${tag}`}
                              className="badge text-bg-light border dense-tag-chip d-inline-flex align-items-center"
                            >
                              <span className="me-1">{tag}</span>
                              <button
                                type="button"
                                className="btn btn-sm p-0 border-0 bg-transparent detection-tag-remove"
                                onClick={() => handleRemoveTagFilter(tag)}
                                aria-label={`Usun filtr tagu ${tag}`}
                              >
                                x
                              </button>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div className="small text-muted mb-2">Brak wybranych tagow.</div>
                      )}

                      <div className="small text-muted mb-1">Tryb dopasowania:</div>
                      <div className="d-flex align-items-center gap-3 mb-1">
                        <div className="form-check form-check-inline mb-0">
                          <input
                            className="form-check-input"
                            type="radio"
                            name="tag-filter-mode"
                            id="tag-filter-mode-or"
                            checked={tagFilterMode === "or"}
                            onChange={() => setTagFilterMode("or")}
                          />
                          <label className="form-check-label small" htmlFor="tag-filter-mode-or">
                            OR
                          </label>
                        </div>
                        <div className="form-check form-check-inline mb-0">
                          <input
                            className="form-check-input"
                            type="radio"
                            name="tag-filter-mode"
                            id="tag-filter-mode-and"
                            checked={tagFilterMode === "and"}
                            onChange={() => setTagFilterMode("and")}
                          />
                          <label className="form-check-label small" htmlFor="tag-filter-mode-and">
                            AND
                          </label>
                        </div>
                      </div>

                      <div className="small text-muted">
                        {tagFilterMode === "or"
                          ? "OR: pokaz detekcje zawierajace dowolny wybrany tag."
                          : "AND: pokaz tylko detekcje zawierajace wszystkie wybrane tagi."}
                      </div>
                    </>
                  )}
                </div>
              )}

              {isNoDetectionsFilterSelected ? (
                <>
                  <div className="form-check mb-2">
                    <input
                      ref={masterNoDetectionsCheckboxRef}
                      className="form-check-input detection-select-checkbox"
                      type="checkbox"
                      checked={areAllVisibleNoDetectionsSelected}
                      onClick={(event) => {
                        event.stopPropagation();
                      }}
                      onChange={(event) => {
                        event.stopPropagation();
                        handleToggleSelectAllNoDetections();
                      }}
                      disabled={
                        selectableNoDetectionImageIds.length === 0 ||
                        deleteModal.isDeleting ||
                        isNoDetectionBulkTagging
                      }
                      aria-label="Zaznacz lub odznacz wszystkie obrazy no_detections"
                    />
                    <label className="form-check-label small ms-1">Select / Deselect All</label>
                  </div>

                  <div className="d-flex gap-2 mb-2">
                    <input
                      className="form-control form-control-sm"
                      type="text"
                      placeholder="Dodaj tag"
                      value={noDetectionBulkTagDraft}
                      onChange={(event) => setNoDetectionBulkTagDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          handleApplyNoDetectionBulkTag();
                        }
                      }}
                      disabled={
                        selectedNoDetectionImageIds.length === 0 ||
                        deleteModal.isDeleting ||
                        isNoDetectionBulkTagging
                      }
                    />
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      onClick={handleApplyNoDetectionBulkTag}
                      disabled={
                        selectedNoDetectionImageIds.length === 0 ||
                        deleteModal.isDeleting ||
                        isNoDetectionBulkTagging ||
                        String(noDetectionBulkTagDraft || "").trim().length === 0
                      }
                    >
                      {isNoDetectionBulkTagging ? "Dodawanie..." : "Dodaj tag"}
                    </button>
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-danger w-100 mb-3"
                    onClick={handleRequestBulkDeleteNoDetectionImages}
                    disabled={
                      selectedNoDetectionImageIds.length === 0 ||
                      deleteModal.isDeleting ||
                      isNoDetectionBulkTagging
                    }
                  >
                    Usun zaznaczone obrazy
                    {selectedNoDetectionImageIds.length > 0
                      ? ` (${selectedNoDetectionImageIds.length})`
                      : ""}
                  </button>
                </>
              ) : (
                <>
                  <div className="form-check mb-2">
                    <input
                      ref={masterDetectionsCheckboxRef}
                      className="form-check-input detection-select-checkbox"
                      type="checkbox"
                      checked={areAllVisibleDetectionsSelected}
                      onClick={(event) => {
                        event.stopPropagation();
                      }}
                      onChange={(event) => {
                        event.stopPropagation();
                        handleToggleSelectAllDetections();
                      }}
                      disabled={
                        selectableDetectionIds.length === 0 ||
                        deleteModal.isDeleting ||
                        isBulkTagging
                      }
                      aria-label="Zaznacz lub odznacz wszystkie detekcje"
                    />
                    <label className="form-check-label small ms-1">Select / Deselect All</label>
                  </div>

                  <div className="d-flex gap-2 mb-2">
                    <input
                      className="form-control form-control-sm"
                      type="text"
                      placeholder="Dodaj tag"
                      value={bulkTagDraft}
                      onChange={(event) => setBulkTagDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          handleApplyBulkTag();
                        }
                      }}
                      disabled={
                        selectedIds.length === 0 || deleteModal.isDeleting || isBulkTagging
                      }
                    />
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-primary"
                      onClick={handleApplyBulkTag}
                      disabled={
                        selectedIds.length === 0 ||
                        deleteModal.isDeleting ||
                        isBulkTagging ||
                        String(bulkTagDraft || "").trim().length === 0
                      }
                    >
                      {isBulkTagging ? "Dodawanie..." : "Dodaj tag"}
                    </button>
                  </div>
                  <div className="d-flex gap-2 mb-2">
                    <select
                      className="form-select form-select-sm"
                      value={exportFormat}
                      onChange={(event) => setExportFormat(event.target.value)}
                      disabled={selectedIds.length === 0 || deleteModal.isDeleting || isBulkTagging}
                    >
                      <option value="json">JSON</option>
                      <option value="csv">CSV</option>
                    </select>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={handleExportSelectedDetections}
                      disabled={selectedIds.length === 0 || deleteModal.isDeleting || isBulkTagging}
                    >
                      Export
                    </button>
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-danger w-100 mb-3"
                    onClick={handleRequestBulkDeleteDetections}
                    disabled={selectedIds.length === 0 || deleteModal.isDeleting || isBulkTagging}
                  >
                    Usun zaznaczone{selectedIds.length > 0 ? ` (${selectedIds.length})` : ""}
                  </button>
                </>
              )}

              <div className="detection-list-scroll" ref={detectionListRef}>
                {isNoDetectionsFilterSelected ? (
                  <>
                    {sortedNoDetectionImages.length === 0 ? (
                      <div className="small text-muted">Brak przeanalizowanych zdjec z wynikiem no_detections.</div>
                    ) : (
                      <div className="list-group dense-detection-list">
                        {sortedNoDetectionImages.map((image, imageIndex) => {
                          const imageId = String(image.image_id || "");
                          const itemKey = `${imageId || "no-id"}|${image.timestamp || "no-ts"}|${imageIndex}`;
                          const isSelectedForGallery =
                            selectedNoDetectionImage?.image_id === image.image_id;
                          const isExpanded = expandedNoDetectionImageId === imageId;
                          const imageName = getFileNameFromPath(image.path);
                          const imageTags = normalizeDetectionTags(image.tags);

                          return (
                            <div
                              key={itemKey}
                              className={`list-group-item list-group-item-action text-start detection-dense-item ${
                                isSelectedForGallery ? "bg-primary-subtle border-primary" : ""
                              }`}
                            >
                              <div
                                className={`detection-dense-row ${isExpanded ? "is-expanded" : ""}`}
                                role="button"
                                tabIndex={0}
                                onClick={() => handleToggleNoDetectionExpand(image)}
                                onKeyDown={(event) => {
                                  if (event.key === "Enter" || event.key === " ") {
                                    event.preventDefault();
                                    handleToggleNoDetectionExpand(image);
                                  }
                                }}
                              >
                                <div className="form-check mb-0">
                                  <input
                                    className="form-check-input detection-select-checkbox"
                                    type="checkbox"
                                    checked={selectedNoDetectionImageIds.includes(imageId)}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                    }}
                                    onChange={(event) => {
                                      event.stopPropagation();
                                      handleToggleNoDetectionSelection(imageId);
                                    }}
                                    aria-label={`Zaznacz obraz ${imageId}`}
                                    disabled={
                                      deleteModal.isDeleting ||
                                      isNoDetectionBulkTagging ||
                                      imageId.length === 0
                                    }
                                  />
                                </div>

                                <div className="detection-dense-id fw-semibold text-truncate">
                                  {imageId || "analysis_image"}
                                </div>

                                <div className="detection-dense-meta">
                                  <span className="badge text-bg-secondary">no_detections</span>
                                </div>

                                <div className="detection-dense-meta text-muted">
                                  {image.resolution || "-"}
                                </div>

                                <div className="detection-dense-tags">
                                  <span className="small text-muted">
                                    lat {formatCoordinate(image.lat, 2)}, lon {formatCoordinate(image.lon, 2)}
                                  </span>
                                </div>

                                <div className="detection-dense-actions">
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-secondary detection-icon-btn"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleOpenAnalysisImage(image);
                                    }}
                                    title="Podglad"
                                    aria-label={`Podglad obrazu ${imageId}`}
                                    disabled={deleteModal.isDeleting || isNoDetectionBulkTagging}
                                  >
                                    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                      <path
                                        fill="currentColor"
                                        d="M16 8s-3-5.5-8-5.5S0 8 0 8s3 5.5 8 5.5S16 8 16 8zM1.173 8a13.133 13.133 0 0 1 1.66-2.043C4.12 4.668 5.88 3.5 8 3.5s3.879 1.168 5.168 2.457A13.133 13.133 0 0 1 14.828 8c-.058.087-.122.183-.195.288-.335.48-.83 1.12-1.465 1.755C11.879 11.332 10.119 12.5 8 12.5s-3.879-1.168-5.168-2.457A13.134 13.134 0 0 1 1.172 8z"
                                      />
                                      <path
                                        fill="currentColor"
                                        d="M8 5.5A2.5 2.5 0 1 0 8 10.5 2.5 2.5 0 0 0 8 5.5zM4.5 8a3.5 3.5 0 1 1 7 0 3.5 3.5 0 0 1-7 0z"
                                      />
                                    </svg>
                                  </button>

                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-danger detection-icon-btn"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleRequestDeleteNoDetectionImage(imageId);
                                    }}
                                    title="Usun obraz"
                                    aria-label={`Usun obraz ${imageId}`}
                                    disabled={
                                      deleteModal.isDeleting ||
                                      isNoDetectionBulkTagging ||
                                      imageId.length === 0
                                    }
                                  >
                                    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                      <path
                                        fill="currentColor"
                                        d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6zM1 3.5A.5.5 0 0 1 1.5 3H4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1h2.5a.5.5 0 0 1 0 1h-.538l-.853 10.66A1 1 0 0 1 12.112 15H3.888a1 1 0 0 1-.997-.84L2.038 4.5H1.5a.5.5 0 0 1-.5-.5zM5 2v1h6V2H5z"
                                      />
                                    </svg>
                                  </button>
                                </div>
                              </div>

                              {isExpanded && (
                                <div className="detection-expand-panel">
                                  <div className="small text-muted mb-1">Szczegoly obrazu</div>
                                  <div className="small mb-1">
                                    <strong>plik:</strong> {imageName || "-"}
                                  </div>
                                  <div className="small mb-1">
                                    <strong>status:</strong> {image.status || NO_DETECTIONS_FILTER}
                                  </div>
                                  <div className="small mb-1">
                                    <strong>resolution:</strong> {image.resolution || "-"}
                                  </div>
                                  <div className="small mb-1">
                                    <strong>lat:</strong> {formatCoordinate(image.lat, 6)}, <strong>lon:</strong>{" "}
                                    {formatCoordinate(image.lon, 6)}
                                  </div>
                                  <div className="small mb-1">
                                    <strong>timestamp:</strong> {image.timestamp || "-"}
                                  </div>
                                  <div className="small mb-2">
                                    <strong>analysis_id:</strong> {image.analysis_id || "-"}
                                  </div>
                                  <div className="small text-muted mb-1">Tagi</div>
                                  {imageTags.length > 0 ? (
                                    <div className="d-flex flex-wrap gap-1 mb-2">
                                      {imageTags.map((tag) => (
                                        <span
                                          key={`${imageId}|tag|${tag}`}
                                          className="badge text-bg-light border dense-tag-chip"
                                        >
                                          {tag}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <div className="small text-muted mb-2">Brak tagow.</div>
                                  )}
                                  <div className="small text-muted text-break">{image.path || ""}</div>

                                  <div className="d-flex justify-content-end mt-2">
                                    <button
                                      type="button"
                                      className="btn btn-sm btn-outline-secondary"
                                      onClick={() => handleOpenAnalysisImage(image)}
                                      disabled={deleteModal.isDeleting || isNoDetectionBulkTagging}
                                    >
                                      Otworz podglad
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </>
                ) : detections.length === 0 ? (
                  <div className="small text-muted">Brak detekcji. Kliknij "Uruchom analize".</div>
                ) : filteredDetections.length === 0 ? (
                  <div className="small text-muted">
                    Brak detekcji dla statusu: {statusFilter}
                    {selectedTagFilters.length > 0
                      ? ` (tagi: ${selectedTagFilters.join(", ")} - ${tagFilterMode.toUpperCase()})`
                      : ""}
                    .
                  </div>
                ) : (
                  <div className="list-group dense-detection-list">
                    {filteredDetections.map((detection, detectionIndex) => {
                      const detectionUniqueId = getDetectionUniqueId(detection);
                      const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                      const isSelected = isSameDetection(selectedDetection, detection);
                      const isExpanded = expandedDetectionId === detection.detection_id;
                      const isHovered = hoveredDetectionId === detectionUniqueId;
                      const statusBadgeClass = getStatusBadgeClass(detection.status);
                      const detectionTags = normalizeDetectionTags(detection.tags);
                      const tagDraftValue = tagDrafts[detection.detection_id] ?? "";
                      const commentText = (detection.comment ?? "").trim();
                      const isEditingThis = editingDetectionId === detection.detection_id;
                      const commentInputValue = isEditingThis ? inputComment : commentText;
                      const hasComment = commentText.length > 0;
                      const resolutionLabel = String(detection?.resolution ?? "").trim();
                      const timestampLabel = String(detection?.timestamp ?? "").trim();
                      const bbox = detection?.bbox ?? {};

                      return (
                        <div
                          key={detectionRenderKey}
                          ref={(node) => {
                            if (node) {
                              detectionItemRefs.current.set(detectionRenderKey, node);
                            } else {
                              detectionItemRefs.current.delete(detectionRenderKey);
                            }
                          }}
                          onMouseEnter={() => setHoveredDetectionId(detectionUniqueId)}
                          onMouseLeave={() => setHoveredDetectionId(null)}
                          className={`list-group-item list-group-item-action text-start detection-dense-item ${
                            isSelected ? "bg-primary-subtle border-primary" : ""
                          }`}
                          style={
                            !isSelected && isHovered
                              ? {
                                  backgroundColor: "rgba(255, 193, 7, 0.08)",
                                  boxShadow: "inset 0 0 0 1px rgba(255, 193, 7, 0.55)",
                                }
                              : undefined
                          }
                        >
                          <div
                            className={`detection-dense-row ${isExpanded ? "is-expanded" : ""}`}
                            role="button"
                            tabIndex={0}
                            onClick={() => handleToggleDetectionExpand(detection)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                handleToggleDetectionExpand(detection);
                              }
                            }}
                          >
                            <div className="form-check mb-0">
                              <input
                                className="form-check-input detection-select-checkbox"
                                type="checkbox"
                                checked={selectedIds.includes(detection.detection_id)}
                                onClick={(event) => {
                                  event.stopPropagation();
                                }}
                                onChange={(event) => {
                                  event.stopPropagation();
                                  handleToggleDetectionSelection(detection.detection_id);
                                }}
                                aria-label={`Zaznacz detekcje ${detection.detection_id}`}
                                disabled={deleteModal.isDeleting || isBulkTagging}
                              />
                            </div>

                            <div className="detection-dense-id fw-semibold text-truncate">
                              {detection.detection_id}
                            </div>

                            <div className="detection-dense-meta">
                              <span className={`badge ${statusBadgeClass}`}>{detection.status}</span>
                            </div>

                            <div className="detection-dense-meta text-muted">
                              {Number(detection.confidence).toFixed(2)}
                            </div>

                            <div className="detection-dense-tags">
                              {detectionTags.length > 0 ? (
                                detectionTags.map((tag) => (
                                  <button
                                    type="button"
                                    key={`${detection.detection_id}|dense|${tag}`}
                                    className={`badge border dense-tag-chip tag-filter-chip-btn ${
                                      selectedTagFilters.includes(tag)
                                        ? "text-bg-primary border-primary is-active"
                                        : "text-bg-light"
                                    }`}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleToggleTagFilter(tag);
                                    }}
                                    title="Kliknij, aby dodac/usunac tag z filtra"
                                    aria-label={`Przelacz filtr tagu ${tag}`}
                                  >
                                    {tag}
                                  </button>
                                ))
                              ) : (
                                <span className="small text-muted">-</span>
                              )}
                            </div>

                            <div className="detection-dense-actions">
                              <button
                                type="button"
                                className="btn btn-sm btn-outline-secondary detection-icon-btn"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleOpenDetectionPreviewModal(detection);
                                }}
                                title="Podglad"
                                aria-label={`Podglad detekcji ${detection.detection_id}`}
                                disabled={deleteModal.isDeleting || isBulkTagging}
                              >
                                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                  <path
                                    fill="currentColor"
                                    d="M16 8s-3-5.5-8-5.5S0 8 0 8s3 5.5 8 5.5S16 8 16 8zM1.173 8a13.133 13.133 0 0 1 1.66-2.043C4.12 4.668 5.88 3.5 8 3.5s3.879 1.168 5.168 2.457A13.133 13.133 0 0 1 14.828 8c-.058.087-.122.183-.195.288-.335.48-.83 1.12-1.465 1.755C11.879 11.332 10.119 12.5 8 12.5s-3.879-1.168-5.168-2.457A13.134 13.134 0 0 1 1.172 8z"
                                  />
                                  <path
                                    fill="currentColor"
                                    d="M8 5.5A2.5 2.5 0 1 0 8 10.5 2.5 2.5 0 0 0 8 5.5zM4.5 8a3.5 3.5 0 1 1 7 0 3.5 3.5 0 0 1-7 0z"
                                  />
                                </svg>
                              </button>

                              <button
                                type="button"
                                className="btn btn-sm btn-outline-danger detection-icon-btn"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleRequestDeleteDetection(detection);
                                }}
                                title="Usun detekcje"
                                aria-label={`Usun detekcje ${detection.detection_id}`}
                                disabled={deleteModal.isDeleting || isBulkTagging}
                              >
                                <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
                                  <path
                                    fill="currentColor"
                                    d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6zM1 3.5A.5.5 0 0 1 1.5 3H4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1h2.5a.5.5 0 0 1 0 1h-.538l-.853 10.66A1 1 0 0 1 12.112 15H3.888a1 1 0 0 1-.997-.84L2.038 4.5H1.5a.5.5 0 0 1-.5-.5zM5 2v1h6V2H5z"
                                  />
                                </svg>
                              </button>
                            </div>
                          </div>

                          {isExpanded && (
                            <div className="detection-expand-panel">
                              <div className="small text-muted mb-1">Metadane</div>
                              <div className="small mb-2">
                                <div><strong>status:</strong> {detection.status || "-"}</div>
                                <div><strong>class:</strong> {detection.class || "-"}</div>
                                <div><strong>confidence:</strong> {Number(detection.confidence).toFixed(2)}</div>
                                <div><strong>resolution:</strong> {resolutionLabel || "-"}</div>
                                <div><strong>analysis_id:</strong> {detection.analysis_id || "-"}</div>
                                <div><strong>timestamp:</strong> {timestampLabel || "-"}</div>
                                <div>
                                  <strong>bbox:</strong>{" "}
                                  x={formatCoordinate(bbox.x, 2)}, y={formatCoordinate(bbox.y, 2)}, w={formatCoordinate(bbox.width, 2)}, h={formatCoordinate(bbox.height, 2)}
                                </div>
                              </div>

                              <div className="small text-muted mb-1">Tagi</div>
                              {detectionTags.length > 0 ? (
                                <div className="d-flex flex-wrap gap-1 mb-2">
                                  {detectionTags.map((tag) => (
                                    <span key={`${detection.detection_id}|${tag}`} className="badge text-bg-light border dense-tag-chip">
                                      <button
                                        type="button"
                                        className={`btn btn-sm p-0 border-0 bg-transparent detection-tag-filter-toggle me-1 ${
                                          selectedTagFilters.includes(tag)
                                            ? "text-primary fw-semibold"
                                            : "text-body"
                                        }`}
                                        onClick={() => handleToggleTagFilter(tag)}
                                        aria-label={`Przelacz filtr tagu ${tag}`}
                                      >
                                        {tag}
                                      </button>
                                      <button
                                        type="button"
                                        className="btn btn-sm p-0 border-0 bg-transparent detection-tag-remove"
                                        onClick={() => handleRemoveTag(detection.detection_id, tag)}
                                        aria-label={`Usun tag ${tag}`}
                                        disabled={deleteModal.isDeleting}
                                      >
                                        x
                                      </button>
                                    </span>
                                  ))}
                                </div>
                              ) : (
                                <div className="small text-muted mb-2">Brak tagow.</div>
                              )}

                              <div className="d-flex gap-2 mb-2">
                                <input
                                  className="form-control form-control-sm"
                                  type="text"
                                  placeholder="Dodaj tag"
                                  value={tagDraftValue}
                                  onChange={(event) =>
                                    setTagDrafts((prev) => ({
                                      ...prev,
                                      [detection.detection_id]: event.target.value,
                                    }))
                                  }
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter") {
                                      event.preventDefault();
                                      handleAddTag(detection.detection_id);
                                    }
                                  }}
                                  disabled={deleteModal.isDeleting}
                                />
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-primary"
                                  onClick={() => handleAddTag(detection.detection_id)}
                                  disabled={deleteModal.isDeleting}
                                >
                                  Dodaj tag
                                </button>
                              </div>

                              <div className="small text-muted mb-1">Komentarz</div>
                              <div className="d-flex gap-2">
                                <input
                                  className="form-control form-control-sm"
                                  type="text"
                                  placeholder="Wpisz komentarz"
                                  value={commentInputValue}
                                  onChange={(event) => {
                                    if (!isEditingThis) {
                                      setEditingDetectionId(detection.detection_id);
                                    }
                                    setInputComment(event.target.value);
                                  }}
                                  onFocus={() => {
                                    if (!isEditingThis) {
                                      setEditingDetectionId(detection.detection_id);
                                      setInputComment(commentText);
                                    }
                                  }}
                                  disabled={deleteModal.isDeleting}
                                />
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-primary"
                                  onClick={() => handleSaveComment(detection.detection_id)}
                                  disabled={deleteModal.isDeleting}
                                >
                                  Zapisz
                                </button>
                                {hasComment && (
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-danger"
                                    onClick={() => handleDeleteComment(detection.detection_id)}
                                    disabled={deleteModal.isDeleting}
                                  >
                                    Usun
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {detectionPreviewModal.isOpen && detectionPreviewModal.detection && (
        <div
          className="confirm-modal-backdrop detection-preview-modal-backdrop"
          role="presentation"
          onClick={handleCloseDetectionPreviewModal}
        >
          <div
            className="confirm-modal-card detection-preview-modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="detection-preview-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="d-flex justify-content-between align-items-start gap-2 mb-2">
              <div>
                <h6 id="detection-preview-title" className="mb-1">
                  Podglad detekcji {detectionPreviewModal.detection.detection_id}
                </h6>
                <div className="small text-muted">Obraz z backendu i metadane detekcji.</div>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={handleCloseDetectionPreviewModal}
              >
                Zamknij
              </button>
            </div>

            {detectionPreviewModal.image?.image_id ? (
              <div className="detection-preview-image-wrap mb-2">
                <div className="detection-preview-image-stage">
                  <img
                    ref={detectionPreviewImageRef}
                    src={getAnalysisImageUrl(detectionPreviewModal.image.image_id)}
                    alt={`Obraz analizy dla ${detectionPreviewModal.detection.detection_id}`}
                    className="detection-preview-image"
                    onLoad={updateDetectionPreviewImageMetrics}
                  />

                  {detectionPreviewModal.showBBoxPreview && detectionPreviewOverlayRect && (
                    <div
                      className="detection-preview-overlay"
                      style={detectionPreviewOverlayRect}
                      aria-label="BBox overlay"
                    />
                  )}
                </div>
              </div>
            ) : (
              <div className="alert alert-warning py-2 mb-2">
                Brak dopasowanego obrazu analizy dla tej detekcji.
              </div>
            )}

            <div className="form-check form-switch mb-2">
              <input
                className="form-check-input"
                type="checkbox"
                id="toggle-detection-preview-bbox"
                checked={detectionPreviewModal.showBBoxPreview}
                onChange={handleToggleDetectionPreviewBBox}
              />
              <label className="form-check-label small" htmlFor="toggle-detection-preview-bbox">
                Pokaz overlay bbox
              </label>
            </div>

            {detectionPreviewModal.showBBoxPreview && !detectionPreviewOverlayRect && (
              <div className="small text-muted mb-2">Brak detekcji</div>
            )}

            <div className="small text-muted detection-preview-meta-grid">
              <div><strong>ID:</strong> {detectionPreviewModal.detection.detection_id}</div>
              <div><strong>analysis_id:</strong> {detectionPreviewModal.detection.analysis_id}</div>
              <div><strong>status:</strong> {detectionPreviewModal.detection.status}</div>
              <div><strong>class:</strong> {detectionPreviewModal.detection.class}</div>
              <div>
                <strong>confidence:</strong> {Number(detectionPreviewModal.detection.confidence).toFixed(2)}
              </div>
              <div>
                <strong>tagi:</strong>{" "}
                {normalizeDetectionTags(detectionPreviewModal.detection.tags).join(", ") || "-"}
              </div>
              <div>
                <strong>bbox:</strong>{" "}
                {JSON.stringify(parseBBoxToMinMax(detectionPreviewModal.detection.bbox) || {})}
              </div>
              <div>
                <strong>obraz_id:</strong> {detectionPreviewModal.image?.image_id || "-"}
              </div>
              <div>
                <strong>obraz_status:</strong> {detectionPreviewModal.image?.status || "-"}
              </div>
              <div>
                <strong>obraz_resolution:</strong> {detectionPreviewModal.image?.resolution || "-"}
              </div>
              <div>
                <strong>obraz_timestamp:</strong> {detectionPreviewModal.image?.timestamp || "-"}
              </div>
              <div className="text-break">
                <strong>obraz_path:</strong> {detectionPreviewModal.image?.path || "-"}
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteModal.targetIds.length > 0 && (
        <div
          className="confirm-modal-backdrop"
          role="presentation"
          onClick={handleCancelDeleteDetection}
        >
          <div
            className="confirm-modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-detection-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h6 id="delete-detection-title" className="mb-2">
              {deleteModal.targetType === "analysis_image" ? "Usunac obraz?" : "Usunac detekcje?"}
            </h6>
            <p className="small text-muted mb-3">
              {deleteModal.targetType === "analysis_image" ? (
                deleteModal.targetIds.length === 1 ? (
                  deleteModal.deleteImages ? (
                    <>
                      Ta operacja usunie obraz <strong>{deleteModal.targetIds[0]}</strong> oraz jego plik z backendu.
                    </>
                  ) : (
                    <>
                      Ta operacja usunie wpis obrazu <strong>{deleteModal.targetIds[0]}</strong>.
                      Plik pozostanie na dysku backendu.
                    </>
                  )
                ) : (
                  deleteModal.deleteImages ? (
                    <>
                      Ta operacja usunie <strong>{deleteModal.targetIds.length}</strong> zaznaczonych obrazow oraz ich pliki z backendu.
                    </>
                  ) : (
                    <>
                      Ta operacja usunie <strong>{deleteModal.targetIds.length}</strong> wpisow obrazow.
                      Pliki pozostana na dysku backendu.
                    </>
                  )
                )
              ) : deleteModal.targetIds.length === 1 ? (
                deleteModal.deleteImages ? (
                  <>
                    Ta operacja usunie detekcje <strong>{deleteModal.targetIds[0]}</strong> oraz
                    sprobuje usunac powiazany plik obrazu z backendu.
                  </>
                ) : (
                  <>
                    Ta operacja usunie detekcje <strong>{deleteModal.targetIds[0]}</strong>.
                    Powiazany obraz pozostanie bez zmian.
                  </>
                )
              ) : (
                deleteModal.deleteImages ? (
                  <>
                    Ta operacja usunie <strong>{deleteModal.targetIds.length}</strong> zaznaczonych
                    detekcji oraz sprobuje usunac powiazane pliki obrazow z backendu.
                  </>
                ) : (
                  <>
                    Ta operacja usunie <strong>{deleteModal.targetIds.length}</strong> zaznaczonych
                    detekcji. Powiazane obrazy pozostana bez zmian.
                  </>
                )
              )}
            </p>

            {(deleteModal.targetType === "detection" || deleteModal.targetType === "analysis_image") && (
              <div className="form-check mb-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="delete-with-images"
                  checked={Boolean(deleteModal.deleteImages)}
                  onChange={(event) =>
                    setDeleteModal((prev) => ({
                      ...prev,
                      deleteImages: event.target.checked,
                    }))
                  }
                  disabled={deleteModal.isDeleting}
                />
                <label className="form-check-label small" htmlFor="delete-with-images">
                  {deleteModal.targetType === "analysis_image"
                    ? "Usun takze pliki obrazow"
                    : "Usun takze powiazane obrazy"}
                </label>
              </div>
            )}

            <div className="d-flex justify-content-end gap-2">
              <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={handleCancelDeleteDetection}
                disabled={deleteModal.isDeleting}
              >
                Anuluj
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger d-flex align-items-center gap-2"
                onClick={handleConfirmDeleteAction}
                disabled={deleteModal.isDeleting}
              >
                {deleteModal.isDeleting && (
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                )}
                <span>{deleteModal.isDeleting ? "Usuwanie..." : "Usun"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
