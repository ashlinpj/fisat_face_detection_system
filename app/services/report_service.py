"""Report service - Report generation and data export"""

import csv
import logging
from datetime import datetime
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ReportService:
    """Generates reports and exports data.

    Dependencies (student_repo, visit_repo) are injected via the constructor.
    """

    def __init__(self, student_repo, visit_repo):
        self.student_repo = student_repo
        self.visit_repo = visit_repo

    def generate_report(self, start_date: str = None, end_date: str = None) -> str:
        """Generate a detailed report for the specified date range"""
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        if end_date is None:
            end_date = start_date

        report = []
        report.append("=" * 60)
        report.append("COLLEGE CANTEEN FACE DETECTION SYSTEM - REPORT")
        report.append("=" * 60)
        report.append(f"Report Period: {start_date} to {end_date}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("-" * 60)

        stats = self.visit_repo.get_daily_statistics(start_date)

        report.append("\nDAILY SUMMARY")
        report.append(f"  Total Visits: {stats['total_visits']}")
        report.append(f"  Unique Visitors: {stats['unique_visitors']}")
        report.append(f"  Unknown Visitors: {stats['unknown_visitors']}")
        report.append(f"  Average Duration: {stats['average_duration_minutes']} minutes")

        students = self.student_repo.get_all()
        report.append(f"\nREGISTERED STUDENTS: {len(students)}")

        logs = self.visit_repo.get_visit_logs(date=start_date)
        report.append(f"\nVISIT LOG ({len(logs)} entries)")
        report.append("-" * 60)
        report.append(f"{'Time':<12} {'Student ID':<15} {'Name':<25} {'Status':<10}")
        report.append("-" * 60)

        for log in logs:
            entry_time = log.get('entry_time', 'N/A')
            if entry_time and 'T' in entry_time:
                entry_time = entry_time.split('T')[1][:8]
            elif entry_time and ' ' in entry_time:
                entry_time = entry_time.split(' ')[1][:8]

            student_id = log.get('student_id', 'Unknown')[:13]
            name = log.get('student_name', 'Unknown')[:23]
            status = "Known" if log.get('is_known') else "Unknown"

            report.append(f"{entry_time:<12} {student_id:<15} {name:<25} {status:<10}")

        report.append("=" * 60)

        return "\n".join(report)

    def export_report_to_file(self, filepath: str, start_date: str = None, end_date: str = None):
        """Export report to a text file"""
        report = self.generate_report(start_date, end_date)
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info("Report exported to: %s", filepath)

    def export_logs_to_csv(self, filepath: str, date: str = None, student_id: str = None):
        """Export visit logs to CSV file"""
        logs = self.visit_repo.get_visit_logs(date=date, student_id=student_id)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Date", "Entry Time", "Exit Time", "Student ID", "Student Name", "Duration (min)", "Status"])

            for log in logs:
                status = "Known" if log.get('is_known') else "Unknown"
                duration = log.get('duration_minutes', '')

                writer.writerow([
                    log['id'],
                    log.get('date', ''),
                    log.get('entry_time', ''),
                    log.get('exit_time', ''),
                    log.get('student_id', ''),
                    log.get('student_name', ''),
                    duration,
                    status,
                ])

        logger.info("Logs exported to: %s", filepath)

    def export_students_to_csv(self, filepath: str):
        """Export student list to CSV file"""
        students = self.student_repo.get_all()

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Student ID", "Name", "Department", "Year", "Registered Date"])

            for student in students:
                created = student.get('created_at', '')[:10] if student.get('created_at') else ''

                writer.writerow([
                    student['id'],
                    student['student_id'],
                    student['name'],
                    student.get('department', ''),
                    student.get('year', ''),
                    created,
                ])

        logger.info("Students exported to: %s", filepath)

    def get_hourly_distribution(self, date: str = None) -> dict:
        """Get hourly distribution of visits"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        logs = self.visit_repo.get_visit_logs(date=date)

        hourly = {i: 0 for i in range(24)}

        for log in logs:
            entry_time = log.get('entry_time', '')
            if entry_time:
                try:
                    if 'T' in entry_time:
                        hour = int(entry_time.split('T')[1][:2])
                    elif ' ' in entry_time:
                        hour = int(entry_time.split(' ')[1][:2])
                    else:
                        continue
                    hourly[hour] += 1
                except (ValueError, IndexError):
                    pass

        return hourly

    def get_peak_hours(self, date: str = None) -> List[Tuple[int, int]]:
        """Get peak hours sorted by visit count"""
        hourly = self.get_hourly_distribution(date)
        sorted_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
        return [(h, c) for h, c in sorted_hours if c > 0]
