import fs from 'fs';
import path from 'path';

const gradeMap = {
  '고1': 'grade1',
  '1학년': 'grade1',
  'grade1': 'grade1',
  '고2': 'grade2',
  '2학년': 'grade2',
  'grade2': 'grade2',
  '고3': 'grade3',
  '3학년': 'grade3',
  'grade3': 'grade3',
  '고등학생': 'grade2'
};

const subjectMap = {
  '국어': 'korean.txt',
  '수학': 'math.txt',
  '영어': 'english.txt',
  '사회': 'social.txt',
  '사회/역사': 'social.txt',
  '역사': 'social.txt',
  '과학': 'science.txt',
  '기타': 'korean.txt'
};

export function loadKnowledgeByGradeSubject(grade, subject) {
  try {
    const gradeDir = gradeMap[grade] || 'grade2';
    const fileName = subjectMap[subject] || 'korean.txt';

    const targetPath = path.join(process.cwd(), 'data', gradeDir, fileName);

    if (fs.existsSync(targetPath)) {
      return fs.readFileSync(targetPath, 'utf-8');
    }

    return loadFallbackKnowledge();
  } catch (error) {
    console.error('학년/과목 지식 데이터 로드 오류:', error);
    return '지식 데이터 로드 실패';
  }
}

export function loadFallbackKnowledge() {
  try {
    const dataDir = path.join(process.cwd(), 'data');

    if (!fs.existsSync(dataDir)) {
      return '지식 데이터 없음';
    }

    const files = fs
      .readdirSync(dataDir)
      .filter((name) => name.endsWith('.txt'))
      .sort();

    if (!files.length) {
      return '지식 데이터 없음';
    }

    let knowledge = '';

    for (const file of files) {
      const fullPath = path.join(dataDir, file);
      const content = fs.readFileSync(fullPath, 'utf-8');
      knowledge += `\n\n=== ${file} ===\n${content}`;
    }

    return knowledge.trim() || '지식 데이터 없음';
  } catch (error) {
    console.error('fallback 지식 데이터 로드 오류:', error);
    return '지식 데이터 로드 실패';
  }
}
