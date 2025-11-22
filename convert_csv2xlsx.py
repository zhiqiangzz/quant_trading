import pandas as pd
import os
import sys
from pathlib import Path
import argparse


def csv_to_xlsx_replace(folder_path, encoding="utf-8", backup=False):
    """
    将指定文件夹中的所有CSV文件转换为XLSX文件并替换原文件

    Parameters:
    - folder_path: 文件夹路径
    - encoding: 文件编码格式
    - backup: 是否创建备份文件
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ 错误: 文件夹 '{folder_path}' 不存在")
        return False

    if not folder.is_dir():
        print(f"❌ 错误: '{folder_path}' 不是一个文件夹")
        return False

    # 查找所有CSV文件
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print(f"📁 在文件夹 '{folder_path}' 中未找到CSV文件")
        return True

    print(f"🔍 找到 {len(csv_files)} 个CSV文件")
    print("=" * 50)

    success_count = 0
    error_count = 0

    for csv_file in csv_files:
        try:
            print(f"🔄 正在处理: {csv_file.name}")

            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding=encoding)

            # 生成XLSX文件名（相同路径，不同扩展名）
            xlsx_file = csv_file.with_suffix(".xlsx")

            if backup and csv_file.exists():
                # 创建备份文件
                backup_file = csv_file.with_suffix(".csv.backup")
                csv_file.rename(backup_file)
                print(f"   💾 已创建备份: {backup_file.name}")

            # 保存为XLSX文件
            df.to_excel(xlsx_file, index=False, engine="openpyxl")

            # 删除原CSV文件
            csv_file.unlink()

            print(f"✅ 转换成功: {csv_file.name} -> {xlsx_file.name}")
            success_count += 1

        except UnicodeDecodeError:
            try:
                # 尝试其他编码
                df = pd.read_csv(csv_file, encoding="gbk")
                xlsx_file = csv_file.with_suffix(".xlsx")

                if backup and csv_file.exists():
                    backup_file = csv_file.with_suffix(".csv.backup")
                    csv_file.rename(backup_file)

                df.to_excel(xlsx_file, index=False, engine="openpyxl")
                csv_file.unlink()

                print(f"✅ 转换成功 (GBK编码): {csv_file.name} -> {xlsx_file.name}")
                success_count += 1

            except Exception as e:
                print(f"❌ 转换失败 {csv_file.name}: {e}")
                error_count += 1

        except Exception as e:
            print(f"❌ 转换失败 {csv_file.name}: {e}")
            error_count += 1

        print("-" * 40)

    print("=" * 50)
    print(f"🎉 转换完成!")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {error_count} 个文件")

    return error_count == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量将CSV文件转换为XLSX文件并替换原文件"
    )
    parser.add_argument("folder", help="要处理的文件夹路径")
    parser.add_argument(
        "--encoding", "-e", default="utf-8", help="CSV文件编码格式 (默认: utf-8)"
    )
    parser.add_argument("--backup", "-b", action="store_true", help="创建CSV备份文件")
    parser.add_argument(
        "--dry-run", "-d", action="store_true", help="模拟运行，不实际转换文件"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🚀 模拟运行模式 (不会实际修改文件)")
        folder = Path(args.folder)
        if folder.exists() and folder.is_dir():
            csv_files = list(folder.glob("*.csv"))
            print(f"📁 找到 {len(csv_files)} 个CSV文件:")
            for csv_file in csv_files:
                xlsx_file = csv_file.with_suffix(".xlsx")
                print(f"   📄 {csv_file.name} -> {xlsx_file.name}")
        return

    print("🚀 CSV转XLSX批量转换工具")
    print("⚠️  注意: 此操作将删除原CSV文件!")

    # 确认操作
    confirm = input("❓ 确定要继续吗? (y/N): ")
    if confirm.lower() not in ["y", "yes"]:
        print("操作已取消")
        return

    # 执行转换
    success = csv_to_xlsx_replace(args.folder, args.encoding, args.backup)

    if success:
        print("✨ 所有文件转换完成!")
    else:
        print("💥 部分文件转换失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
