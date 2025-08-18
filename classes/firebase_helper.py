from datetime import datetime, timezone
import json
from google.protobuf.timestamp_pb2 import Timestamp
from time import time
import uuid
from typing import Type, TypeVar, Union
from firebase_admin import credentials, storage, initialize_app, firestore
from pydantic import BaseModel
import rich

T = TypeVar("T")

service_account_info = {
    "type": "service_account",
    "project_id": "slash-and-burn",
    "private_key_id": "b178c34eaf0153c784b5341b633144f4fa048063",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCP5PghCor5V2Aj\nIzvGLxyOZ32KacMN2vCnSOPmbURBGiQQrMWvn3zbqJknSG5fIbzThaJ/A+VtzlEa\n+ZhMnf4avUdGhq0b0/QvQUzHiTC33iAhepofMxFJ+1P5NirfulN4ulXkgL1Nf15Y\nxuFGS/BB2E/Vc7Ms18O796s0wb3noQxHwRj8tNI1jW0Zm6S4J5Gv6dATm8EClsjn\n8vjZMFZDoR0ImS3bp3xikPGvBcXmEoHnlzMVGFAXnxfALu5i2sDitQUj8MB/e3AU\n25KgxDeeDq6pYz7+cvrQGQes9Ueb0lEc35SIxFSsLNGR5gIuoFlmwrYnUAOkkVbf\nNvTtLH0bAgMBAAECggEAM9/1bh/5YxOTZod8tKFmV5ZKpXwkZBnSmVRHSNqKeTfd\n2Z8Xs405O41IDWfo1mX4x37NSqISc6gmCCYEOFba9TPYfr3tqXMbfG+7qNG2HQs4\npSkoZ+gGqxeuudDD2x02m3b9oR+iX050Kmgkba3Rw0Mi4M5RwXBVsa2DbMUN2/Ni\ncYmQUV1fgGm43rxin9ozJCWVYFc2BNTwz3CfX7Hh9kU01fNFTZXEmTySd9thAL0F\nGyVb7wZvPLi2ip9zs1Kqu8r/7D3Mzlf97+uGLGr6wG2SrakvnBnZxvxNEgzvlKTP\nj7yR7C6W2/UpOy4sVrQEM6VogPRAwzhKqGF1XgMMmQKBgQDAjKX13AZ62KQrJuLe\n1IGGpSMhafhVv7PeMhy89dJ0zpzfmqj4lQ2zaMNjYOqTeTvkt1AYwF1uVDCfFQjK\nx8jvNt8rLZK3bAw9p0nw7hqVVm0RlU2STXOwEHk6jtRRKg5Qa1OO99MIQrhxMUKm\ngX5o6r6/UdgZHVClJkPzKNypswKBgQC/T9CbtX8Kjy3XiQKJJ6W6nmpT0gVB/Ttv\nKCI2iDZJbk7E9KAP6Au4uQnfrGmweIeEoOv7jJZdNq5/6emX6KV0ZWSBqxvAAxHh\nYusPR/+VQFoVUK1VTTnvFA1NVVLblpiRNW7FKJJOF0TJLDfP7fIIPTJalyhE22Yk\nbw6FO1Pa+QKBgG1Zswq8qQVtMVa8X82Su/ieoiPgzcLM+zZuGToLFTl2+UpXyCxc\nYgqIraYrrSyRhj0vChL0dFsq/u3pgTPAYFHSRM19tTvr5cvBzNFCN+Cl041P0F0N\nFW1g/agO428wxi5PtYWVIsknMx49jo+HLSYiYq++qE4jAuC3qZFXnHHXAoGAaPUt\n+q85Up64xL37MSoaR2cv6GqZzdlTaSl4k7hpKCInfvDCe9ePzrldzGP33ARUPRRY\nQzqfJ+afF7hKGrhdRZ125ZjtcYt9nMy7LzgN5WFXysfnQJxMw3iZz6qW+bgGXewV\nqH8YvVUQvNnQiPf/SDviy0aQpi5EPIrdSYUB/WkCgYEAryXvk1VYLlbGcdCamiQA\nWGAnfD10w6IOI3ThHhDNnLReD0THYFgpDudapdno4PULShSWdSjUaTacepQaABzB\nDTigYR9i5tz+f+AVWkvMZyK0EzjQMIB+8POJKgZH9+IVBbyqg8J4mIp2MxpQ5+I/\nLkxF0IcvTnEHXe6RbEZqeMM=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-fbsvc@slash-and-burn.iam.gserviceaccount.com",
    "client_id": "105345884374884829562",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40slash-and-burn.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}


class Firebase:
    def __init__(
        self,
        credentials_filepath: str,
        use_storage=False,
        use_firestore=False,
        storage_bucket="",
    ):
        # print("-----credentials_filepath---------")
        # print(credentials_filepath)
        self.cred = credentials.Certificate(service_account_info)
        # self.cred = credentials.Certificate(credentials_filepath)
        # rich.print(self.cred.get_credential().project_id)

        # with open("credentials.json", "r", encoding="utf-8") as f:
        #     data = json.load(f)
        #     rich.print(data)
        initialize_app(
            self.cred,
            {"storageBucket": storage_bucket} if use_storage else None,
        )
        self.db = firestore.client() if use_firestore else None

    #! STORAGE
    def upload_storage(
        self, file_path: str, destination_blob_name: str, make_public=True
    ):
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        if make_public:
            blob.make_public()

        return blob.public_url

    #! FIRESTORE
    def read_firestore(self, doc_path: str, data_type: Type[T]) -> T:
        if self.db is None:
            raise ValueError("Firestore is not initialized.")

        doc_ref = self.db.document(doc_path)
        data = doc_ref.get().to_dict()

        if data is None:
            return None

        return data_type(**data)

    def write_firestore(self, doc_path: str, data: Union[BaseModel, dict]):
        if self.db is None:
            raise ValueError("Firestore is not initialized.")

        doc_ref = self.db.document(doc_path)
        if isinstance(data, BaseModel):
            doc_ref.set(data.model_dump(exclude_unset=True))
        else:
            doc_ref.set(data)

    def update_firestore(self, doc_path: str, data: Union[BaseModel, dict]):
        if self.db is None:
            raise ValueError("Firestore is not initialized.")

        doc_ref = self.db.document(doc_path)
        if isinstance(data, BaseModel):
            doc_ref.update(data.model_dump(exclude_unset=True))
        else:
            doc_ref.update(data)

    def exists_firestore(self, doc_path: str) -> bool:
        if self.db is None:
            raise ValueError("Firestore is not initialized.")

        doc_ref = self.db.document(doc_path)
        return doc_ref.get().exists

    #! SERVER TIMESTAMP
    def timestamp(self):
        return datetime.now(timezone.utc)

    #! UUID
    def uuid(self) -> str:
        return uuid.uuid4().hex
