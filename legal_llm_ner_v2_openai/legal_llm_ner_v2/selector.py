from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple
KEYWORDS={
"BEHAVIOR":["vượt","đi ngược","không đội","chở quá","quá tải","dừng xe","đỗ xe","đi vào đường cấm","nồng độ cồn","ma túy","không chấp hành","hiệu lệnh","đua xe","lạng lách","đánh võng","chở hàng","vận chuyển","gây tai nạn"],
"VEHICLE":["xe mô tô","xe gắn máy","xe ô tô","ô tô tải","xe tải","xe khách","xe buýt","xe máy chuyên dùng","xe đạp điện","xe thô sơ","rơ moóc","sơ mi rơ moóc","máy kéo","xe taxi","xe hợp đồng"],
"ACTOR":["người điều khiển xe","người lái xe","chủ xe","người đi bộ","người ngồi trên xe","học viên lái xe","giáo viên dạy lái","cơ sở đào tạo lái xe","trung tâm sát hạch","cảnh sát giao thông","cơ quan đăng ký xe","đơn vị kinh doanh vận tải"],
"INFRASTRUCTURE":["đường cao tốc","làn đường","phần đường","cầu","hầm","đường ngang","bến xe","trạm thu phí","đèn tín hiệu","biển báo","vạch kẻ đường","dải phân cách","sân sát hạch","camera giám sát","thiết bị giám sát hành trình","phần mềm mô phỏng","hệ thống quản lý","khu đông dân cư"],
"DOCUMENT":["giấy phép lái xe","giấy đăng ký xe","chứng nhận kiểm định","bảo hiểm trách nhiệm dân sự","căn cước công dân","hộ chiếu","giấy khai đăng ký xe","biên bản vi phạm","chứng chỉ","phù hiệu xe","giấy phép kinh doanh vận tải"],
"VEHICLE_CONDITION_OR_EQUIPMENT":["gương chiếu hậu","kết cấu xe","màu sơn","đèn chiếu sáng","còi","lốp","khí thải","hệ thống phanh","biển số","che lấp","camera hành trình","thiết bị giám sát hành trình","không có","không bảo đảm","không đạt chuẩn"],
"CONDITION":["đủ tuổi","chưa đủ tuổi","đủ sức khỏe","tiêu chuẩn sức khỏe","có giấy phép","không có giấy phép","giấy phép lái xe phù hợp","hoàn thành chương trình đào tạo","có nồng độ cồn","đáp ứng","điều kiện"],
}
def candidate_score(sentence: Dict[str,Any]) -> Tuple[int,List[str]]:
    text=(sentence.get("text") or "").lower(); ctx=(sentence.get("context_text") or sentence.get("path_text") or "").lower()
    labels=[]; score=0
    for lab,kws in KEYWORDS.items():
        text_hit=False; ctx_hit=False
        for kw in kws:
            if kw in text:
                text_hit=True; score += 2
            elif kw in ctx:
                ctx_hit=True
        if text_hit or (ctx_hit and score > 0): labels.append(lab)
    if re.search(r"\bxe\b", text): score+=1; labels.append("VEHICLE")
    if re.search(r"\bgiấy\b|\bchứng nhận\b|\bbiên bản\b", text): score+=1; labels.append("DOCUMENT")
    return score, sorted(set(labels))
def select_candidates(sentences: List[Dict[str,Any]], min_score:int=1) -> List[Dict[str,Any]]:
    out=[]
    for s in sentences:
        score, labs = candidate_score(s)
        if score>=min_score:
            r=dict(s); r["_candidate_score"]=score; r["_candidate_labels"]=labs; out.append(r)
    out.sort(key=lambda x:x.get("_candidate_score",0), reverse=True)
    return out
