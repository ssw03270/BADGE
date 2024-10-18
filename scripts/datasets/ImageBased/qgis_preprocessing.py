import os
import csv
from qgis.core import *
from qgis.utils import iface
from PyQt5.QtCore import QSize, QSizeF, QVariant
from PyQt5.QtGui import QColor
import processing
import pickle
from tqdm import tqdm
import gc  # 파일 상단에 추가

# Paths and variables
csv_path = 'Z:/iiixr-drive/Projects/2024_City_Team/CityBoundaries/500Cities_City_11082016/city_bounding_boxes.csv'  # 실제 경로로 변경하세요
output_dir = 'D:/City_Team/Outputs'    # 실제 경로로 변경하세요

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 생성

# Parameters
m = 1000  # Size of each grid cell in meters

# Read the CSV file
cities = []
with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader, desc="Reading CSV"):
        cities.append({
            'city_name': row['city_name'],
            'state': row['state'],
            'lon_min': float(row['x_min']),
            'lon_max': float(row['x_max']),
            'lat_min': float(row['y_min']),
            'lat_max': float(row['y_max'])
        })

# Coordinate transformation from EPSG:4326 to EPSG:3857
source_crs = QgsCoordinateReferenceSystem('EPSG:4326')
target_crs = QgsCoordinateReferenceSystem('EPSG:3857')
transform_context = QgsProject.instance().transformContext()
coord_transform = QgsCoordinateTransform(source_crs, target_crs, transform_context)

# Retrieve the 'CityBoundaries' layer
project = QgsProject.instance()
city_boundaries_layers = project.mapLayersByName('CityBoundaries')
if not city_boundaries_layers:
    print("CityBoundaries layer not found in the project.")
    exit()
else:
    city_boundaries_layer = city_boundaries_layers[0]  # Use the existing layer

# Ensure 'CityBoundaries' layer is in EPSG:3857
if city_boundaries_layer.crs().authid() != 'EPSG:3857':
    print("Reprojecting 'CityBoundaries' layer to EPSG:3857.")
    city_boundaries_layer = processing.run("native:reprojectlayer", {
        'INPUT': city_boundaries_layer,
        'TARGET_CRS': target_crs,
        'OUTPUT': 'memory:'
    })['OUTPUT']

# Retrieve the existing layers from the project

# Google Satellite Layer (Already in EPSG:3857)
google_satellite_layers = project.mapLayersByName('Google Satellite')
if not google_satellite_layers:
    print("Google Satellite layer not found in the project.")
    exit()
else:
    google_satellite_layer = google_satellite_layers[0]  # Use the existing layer

# Set up the layout
manager = project.layoutManager()
layout_name = 'City_Patches'

# Remove existing layout with the same name, if it exists
existing_layout = manager.layoutByName(layout_name)
if existing_layout:
    manager.removeLayout(existing_layout)

# Create a new layout
layout = QgsPrintLayout(project)
layout.initializeDefaults()
layout.setName(layout_name)
manager.addLayout(layout)

# Adjust the map item to be square
map_item = QgsLayoutItemMap(layout)
map_item.setRect(0, 0, 200, 200)  # Set to a square size in millimeters
layout.addLayoutItem(map_item)

# Set the layout page size to match the map item
page = layout.pageCollection().pages()[0]
page.setPageSize(QgsLayoutSize(200, 200, QgsUnitTypes.LayoutMillimeters))

# Set the map item's CRS to EPSG:3857
map_item.setCrs(target_crs)

# Desired output image dimensions in pixels
output_width = 1000
output_height = 1000

# Export settings
exporter = QgsLayoutExporter(layout)
settings = QgsLayoutExporter.ImageExportSettings()
settings.imageSize = QSize(output_width, output_height)
settings.pngCompressionLevel = 0  # 0 is best quality

# Set map background color to white
map_item.setBackgroundColor(QColor(255, 255, 255))  # Set background to white

# 프로젝트에서 모든 레이어 그룹 가져기
root = QgsProject.instance().layerTreeRoot()

# 레이어나 그룹의 활성화 상태를 변경하는 함수 추가
def set_visibility(node, visible):
    if node:
        node.setItemVisibilityChecked(visible)
        # print(f"'{node.name()}'의 가시성이 {visible}로 설정되었습니다.")
    else:
        print(f"노드를 찾을 수 없습니다.")

# 각 도시에 대한 처리
for city in cities:
    city_name = city['city_name']
    state = city['state']
    lon_min = city['lon_min']
    lon_max = city['lon_max']
    lat_min = city['lat_min']
    lat_max = city['lat_max']

    print(f"Processing city: {city_name}, State: {state}")

    # 해당 주의 그룹 찾기
    state_group = next((group for group in root.children() if group.name() == state), None)
    if not state_group:
        print(f"{state} 그룹을 찾을 수 없습니다. {city_name} 건너뛰기.")
        continue

    # 주 그룹 활성화
    set_visibility(state_group, True)

    # 주 그룹 내에서 Road Network와 Building Footprint 레이어 찾기
    road_network_layer_node = next((layer for layer in state_group.children() if layer.name() == 'Road Network'), None)
    building_footprint_layer_node = next((layer for layer in state_group.children() if layer.name() == 'Building Footprint'), None)

    if not road_network_layer_node or not building_footprint_layer_node:
        print(f"{state}의 필요한 레이어를 찾을 수 없습니다. {city_name} 건너뛰기.")
        set_visibility(state_group, False)  # 주 그룹 비활성화
        continue

    # 필요한 레이어 활성화
    set_visibility(road_network_layer_node, True)
    set_visibility(building_footprint_layer_node, True)

    # 레이어 재투영 (필요한 경우)
    for layer_node, layer_name in [(road_network_layer_node, 'Road Network'), (building_footprint_layer_node, 'Building Footprint')]:
        layer = layer_node.layer()  # 실제 QgsVectorLayer 객체 가져오기
        reprojected_layer_name = f"{layer.name()}_3857"

        # 이미 변환된 레이어가 있는지 확인
        existing_reprojected_layer_node = next((l for l in state_group.children() if l.name() == reprojected_layer_name), None)

        if existing_reprojected_layer_node is None:
            if layer.crs().authid() != 'EPSG:3857':
                # print(f"'{state}' {layer_name} 레이어를 EPSG:3857로 재투영 중.")

                # 재투영된 레이어를 저장할 디렉토리 경로
                reprojected_layer_dir = os.path.join(output_dir, "reprojected_layer")
                # 디렉토리가 존재하지 않으면 생성
                os.makedirs(reprojected_layer_dir, exist_ok=True)

                # 재투영된 레이어를 파일로 저장
                reprojected_layer_path = os.path.join(reprojected_layer_dir, f"{state}_{reprojected_layer_name}.gpkg")
                processing.run("native:reprojectlayer", {
                    'INPUT': layer,
                    'TARGET_CRS': target_crs,
                    'OUTPUT': reprojected_layer_path  # 파일로 저장
                })

                # 재투영된 레이어를 다시 로드
                reprojected_layer = QgsVectorLayer(reprojected_layer_path, reprojected_layer_name, "ogr")

                # 레이어 이름 설정
                reprojected_layer.setName(reprojected_layer_name)

                # 스타일 적용
                reprojected_layer.setRenderer(layer.renderer().clone())

                # **프로젝트에 레이어 추가**
                QgsProject.instance().addMapLayer(reprojected_layer, addToLegend=True)  # addToLegend=True로 설정

                # 레이어 트리에 새 레이어 추가
                new_layer_node = QgsLayerTreeLayer(reprojected_layer)
                state_group.addChildNode(new_layer_node)

                # 변수 업데이트
                if layer_name == 'Road Network':
                    road_network_layer_node = new_layer_node
                else:
                    building_footprint_layer_node = new_layer_node

            else:
                print(f"'{state}' {layer_name} 레이어는 이미 EPSG:3857 좌표계입니다.")
                # 레이어 이름 업데이트
                layer.setName(reprojected_layer_name)
                road_network_layer_node = layer_node if layer_name == 'Road Network' else road_network_layer_node
                building_footprint_layer_node = layer_node if layer_name == 'Building Footprint' else building_footprint_layer_node

        else:
            # 이미 존재하는 재투영된 레이어 사용
            print(f"'{state}' {reprojected_layer_name} 레이어가 이미 존재합니다.")
            if layer_name == 'Road Network':
                road_network_layer_node = existing_reprojected_layer_node
            else:
                building_footprint_layer_node = existing_reprojected_layer_node

    # QGIS 프로젝트 저장
    QgsProject.instance().write()

    # Apply style
    road_network_layer_node.layer().setRenderer(road_network_layer_node.layer().renderer().clone())
    building_footprint_layer_node.layer().setRenderer(building_footprint_layer_node.layer().renderer().clone())

    # print(f"BBox (Lat/Lon): lon_min={lon_min}, lon_max={lon_max}, lat_min={lat_min}, lat_max={lat_max}")

    # 도시 이름으로 폴더 생성
    city_output_dir = os.path.join(output_dir, city_name)
    os.makedirs(city_output_dir, exist_ok=True)

    # Transform BBox coordinates from Lat/Lon to EPSG:3857
    bottom_left = QgsPointXY(lon_min, lat_min)
    top_right = QgsPointXY(lon_max, lat_max)

    # Transform points
    bottom_left_transformed = coord_transform.transform(bottom_left)
    top_right_transformed = coord_transform.transform(top_right)

    x_min = bottom_left_transformed.x()
    y_min = bottom_left_transformed.y()
    x_max = top_right_transformed.x()
    y_max = top_right_transformed.y()

    print(f"BBox (EPSG:3857): x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    # Filter the city boundaries to get the boundary for the current city
    expression = QgsExpression(f"\"NAME\" = '{city_name}'")
    request = QgsFeatureRequest(expression)
    city_boundary_feature = next(city_boundaries_layer.getFeatures(request), None)
    if city_boundary_feature is None:
        print(f"{city_name} boundary not found in CityBoundaries layer.")

        set_visibility(road_network_layer_node, False)
        set_visibility(building_footprint_layer_node, False)
        set_visibility(state_group, False)  # 주 그룹 비활성화
        
        continue  # Skip this city if boundary not found

    city_boundary_geom = city_boundary_feature.geometry()

    # Calculate number of grid cells in x and y directions
    x_count = int((x_max - x_min) / m) + 1
    y_count = int((y_max - y_min) / m) + 1

    # 전체 그리드 셀 수 계산
    total_cells = x_count * y_count

    # tqdm을 사용한 단일 for문
    for idx in tqdm(range(total_cells), desc=f"Processing {city_name}", unit="cell"):        
        # Check if all required files exist
        required_files = [
            f"{city_name.replace(' ', '')}_grid_{idx}_Google_Satellite.png",
            f"{city_name.replace(' ', '')}_grid_{idx}_Road_Network.png",
            f"{city_name.replace(' ', '')}_grid_{idx}_Building_Footprint.png",
            f"{city_name.replace(' ', '')}_grid_{idx}_info.pkl",
        ]
        if all(os.path.exists(os.path.join(city_output_dir, file)) for file in required_files):
            continue  # Skip if all required files exist

        # i와 j 계산
        i = idx // y_count
        j = idx % y_count

        # 그리드 셀 좌표 계산
        x_start = x_min + i * m
        x_end = x_start + m
        y_start = y_min + j * m
        y_end = y_start + m

        # 중심점 계산
        x_center = (x_start + x_end) / 2
        y_center = (y_start + y_end) / 2

        # 그리드 셀 지오메트리 생성
        grid_rect = QgsRectangle(x_start, y_start, x_end, y_end)
        grid_geom = QgsGeometry.fromRect(grid_rect)

        # 도시 경계와 교차하는지 확인
        if not grid_geom.intersects(city_boundary_geom):
            continue  # 도시 경계와 교차하지 않으면 건너뛰기

        # 건물 경계와 교차하는지 확인
        building_intersects = False
        request = QgsFeatureRequest().setFilterRect(grid_rect)  # 그리드 셀 범위로 필터링
        for building_feature in building_footprint_layer_node.layer().getFeatures(request):
            if grid_geom.intersects(building_feature.geometry()):
                building_intersects = True
                break

        if not building_intersects:
            continue  # 건물 경계와 교차하지 않으면 건너뛰기

        # print(f"Processing grid cell {idx} for city {city_name}")

        # Zoom map to the grid cell extent
        map_item.zoomToExtent(grid_rect)
        map_item.setExtent(grid_rect)

        # Layers to export
        layers_to_export = [
            ('Google Satellite', google_satellite_layer),
            ('Road Network', road_network_layer_node.layer()),
            ('Building Footprint', building_footprint_layer_node.layer())
        ]

        for layer_name, layer in layers_to_export:
            # Set the map item's layers to include only the current layer
            map_item.setLayers([layer])

            # Ensure the output filename has the correct extension
            output_filename = f"{city_name.replace(' ', '')}_grid_{idx}_{layer_name.replace(' ', '_')}.png"
            output_path = os.path.join(city_output_dir, output_filename)  # 도시 폴더에 저장

            # Export the layout to an image
            result = exporter.exportToImage(output_path, settings)

            if result != QgsLayoutExporter.Success:
                print(f"오류: {city_name}의 그리드 {idx} ({layer_name}) 이미지 내보내기 실패")
            # else 문은 제거됨

        # 그리드 정보를 pkl 파일로 저장
        grid_info = {
            'grid_index': idx,
            'x_start': x_start,
            'x_end': x_end,
            'y_start': y_start,
            'y_end': y_end,
            'x_center': x_center,
            'y_center': y_center,
            'area': grid_geom.area()
        }
        pkl_filename = os.path.join(city_output_dir, f"{city_name.replace(' ', '')}_grid_{idx}_info.pkl")
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(grid_info, pkl_file)

        # Clear variables to free memory
        del grid_geom

    # 처리가 끝난 후 레이어 비활성화
    set_visibility(road_network_layer_node, False)
    set_visibility(building_footprint_layer_node, False)
    set_visibility(state_group, False)  # 주 그룹 비활성화

    print(f"Completed processing for city: {city_name}")
    
    # 대용량 객체 명시적 해제
    del city_boundary_geom
    del city_boundary_feature
    
    # 주기적으로 가비지 컬렉션 실행 (예: 10개 도시마다)
    if cities.index(city) % 10 == 0:
        gc.collect()

# Remove the layout after exporting images
manager.removeLayout(layout)
print("Layout removed after exporting images.")

# 모든 처리가 끝난 후 QGIS 프로젝트 다시 저장
QgsProject.instance().write()
