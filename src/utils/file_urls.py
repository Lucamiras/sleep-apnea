def get_download_urls() -> tuple:
    """
    This is a convenience function to keep the main.py file clean. It returns two lists:
    1. edf_urls
    2. rml_urls
    :returns: None
    """

    edf_urls = [
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1ee&path=/V3/APNEA_EDF/00000995-100507/00000995-100507[002].edf&fileName=00000995-100507%5B002%5D.edf",
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1ef&path=/V3/APNEA_EDF/00000995-100507/00000995-100507[001].edf&fileName=00000995-100507%5B001%5D.edf",
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1ed&path=/V3/APNEA_EDF/00000995-100507/00000995-100507[003].edf&fileName=00000995-100507%5B003%5D.edf",
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1eb&path=/V3/APNEA_EDF/00000995-100507/00000995-100507[005].edf&fileName=00000995-100507%5B005%5D.edf",
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842cf1ec&path=/V3/APNEA_EDF/00000995-100507/00000995-100507[004].edf&fileName=00000995-100507%5B004%5D.edf",
    ]

    rml_urls = [
        "https://download.scidb.cn/download?fileId=61a0ca3f89f14b48842ced8b&path=/V3/APNEA_RML_clean/00000995-100507.rml&fileName=00000995-100507.rml",
    ]

    return edf_urls, rml_urls
