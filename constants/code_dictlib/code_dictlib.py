###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2019年1月6日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################


class CodeForAdd():
    permission_code_demo = 'const-string v0, \"android.permission.INTERNET\"'
    
    api_code_demo = '    invoke-virtual {p0}, Landroid/content/Context;->getPackageManager()Landroid/content/pm/PackageManager;'
    +'\n\n' + 'move-result-object v0'
    
    log_demo = '    const-string v1, \"li\"' 
    +'    const-string v2, \"Sensitive infomation\"'
    +'    invoke-static {v1, v2}, Landroid/util/Log;->v(Ljava/lang/String;Ljava/lang/String;)I'

    api_code_dict = {
        'android/app/Activity;->startActivity':'1.android_app_activity_startActivity',
        'java/net/HttpURLConnection':'2.java_net_HttpURLConnection',
        'android/telephony/TelephonyManager;->getDeviceId':'3.android_telephony_TelephonyManager_getDeviceId',
        'android/location/LocationManager;->getLastKnownLocation':'4',
        'android/net/ConnectivityManager;->getActiveNetworkInfo':'5.android_net_ConnectivityManager_getActiveNetworkInfo',
        'android/net/wifi/WifiManager;->getConnectionInfo':'6.android_net_wifi_wifimanager_getconnectioninfo',
        'android/webkit/WebView':'7',
        'android/content/Context;->startActivity':'8.android_content_Context_startActivity',
        'java/lang/Runtime;->exec':'9.java_lang_Runtime_exec',
        'android/net/wifi/WifiManager;->setWifiEnabled':'10.android_net_wifi_WifiManager_setWifiEnabled',
        'android/telephony/TelephonyManager;->getCellLocation':'11',
        'android/app/NotificationManager;->notify':'12',
        'android/app/ActivityManager;->getRunningTasks':'13',
        'android/content/ContentResolver;->query':'14',
        'android/os/Vibrator;->vibrate':'15',
        'android/accounts/AccountManager;->getAccounts':'16',
        'android/os/PowerManager$WakeLock;->acquire':'17',
        'android/telephony/SmsManager;->sendTextMessage':'18',
        'android/location/LocationManager;->getBestProvider':'19',
        'android/location/LocationManager;->isProviderEnabled':'20',
        'android/media/MediaPlayer;->stop':'21',
        'android/telephony/TelephonyManager;->listen':'22',
        'org/apache/http/impl/client/DefaultHttpClient':'23',
        'java/net/ServerSocket':'24',
        'android/media/MediaPlayer;->start':'25',
        'android/media/MediaRecorder;->setAudioSource':'26',
        'android/net/ConnectivityManager;->getAllNetworkInfo':'27',
        'java/net/URL;->openConnection':'28',
        'android/app/KeyguardManager$KeyguardLock;->disableKeyguard':'29',
        'android/telephony/TelephonyManager;->getLine1Number':'30',
        'android/location/LocationManager;->requestLocationUpdates':'31',
        'java/net/Socket':'32',
        'android/os/Vibrator;->cancel':'33',
        'android/content/Context;->startService':'34',
        'android/net/ConnectivityManager;->getNetworkInfo':'35',
        'android/content/ContentResolver;->openFileDescriptor':'36',
        'android/hardware/Camera;->open':'37',
        'java/net/URL;->openStream':'38',
        'android/net/wifi/WifiManager;->startScan':'39',
        'android/telephony/TelephonyManager;->getSubscriberId':'40',
        'android/location/LocationManager;->addGpsStatusListener':'41',
        'android/net/wifi/WifiManager;->isWifiEnabled':'42',
        'android/media/AudioRecord':'43',
        'android/media/AudioManager;->setMode':'44',
        'android/content/ContentResolver;->openInputStream':'45',
        'java/net/DatagramSocket':'46',
        'android/content/ContentResolver;->getCurrentSync':'47',
        'android/media/AudioManager;->isBluetoothA2dpOn':'48',
        'android/bluetooth/BluetoothAdapter;->disable':'49',
        'android/content/ContentResolver;->setMasterSyncAutomatically':'50',
        'android/net/wifi/WifiManager;->getWifiState':'51',
        'android/bluetooth/BluetoothAdapter;->getScanMode':'52',
        'android/content/ContentResolver;->getPeriodicSyncs':'53',
        'android/app/ActivityManager;->killBackgroundProcesses':'54',
        'android/provider/Settings$System;->putInt':'55',
        'android/content/pm/PackageManager;->setComponentEnabledSetting':'56',
        'android/telephony/gsm/SmsManager;->sendTextMessage':'57',
        'android/accounts/AccountManager;->invalidateAuthToken':'58',
        'android/accounts/AccountManager;->getAuthToken':'59',
        'android/content/Context;->sendBroadcast':'60',
        'android/telephony/TelephonyManager;->getNeighboringCellInfo':'61',
        'android/accounts/AccountManager;->addAccount':'62',
        'android/accounts/AccountManager;->addOnAccountsUpdatedListener':'63',
        'android/content/ContentResolver;->getMasterSyncAutomatically':'64',
        'android/app/ActivityManager;->restartPackage':'65',
        'android/content/Context;->removeStickyBroadcast':'66',
        'android/content/Context;->clearWallpaper':'67',
        'android/os/PowerManager;->goToSleep':'68',
        'android/net/ConnectivityManager;->requestRouteToHost':'69',
        'android/media/RingtoneManager;->setActualDefaultRingtoneUri':'70',
        'android/net/wifi/WifiManager$WifiLock;->release':'71',
        'android/content/ContentResolver;->getSyncAutomatically':'72',
        'android/accounts/AccountManager;->addAccountExplicitly':'73',
        'android/content/ContentResolver;->setSyncAutomatically':'74',
        'android/app/Activity;->setWallpaper':'75',
        'android/app/WallpaperManager;->setBitmap':'76',
        'android/net/wifi/WifiManager$WifiLock;->acquire':'77',
        'android/telephony/TelephonyManager;->getSimSerialNumber':'78',
        'android/bluetooth/BluetoothDevice;->createRfcommSocketToServiceRecord':'79',
        'java/net/URL;->getContent':'80',
        'android/bluetooth/BluetoothAdapter;->cancelDiscovery':'81',
        'android/provider/Contacts$People;->createPersonInMyContactsGroup':'82',
        'android/provider/Settings$System;->putString':'83',
        'android/os/PowerManager$WakeLock;->release':'84',
        'android/provider/Settings$Secure;->putString':'85',
        'android/content/Context;->setWallpaper':'86',
        'android/net/ConnectivityManager;->startUsingNetworkFeature':'87',
        'android/speech/SpeechRecognizer;->cancel':'88',
        'android/app/backup/BackupManager;->dataChanged':'89',
        'android/provider/UserDictionary$Words;->addWord':'90',
        'android/net/wifi/WifiManager$MulticastLock;->release':'91',
        'android/net/wifi/WifiManager;->getWifiApState':'92',
        'android/net/wifi/WifiManager;->addOrUpdateNetwork':'93',
        'android/telephony/gsm/SmsManager;->sendMultipartTextMessage':'94',
        'android/app/WallpaperManager;->setResource':'95',
        'java/net/URLConnection;->getInputStream':'96',
        'java/net/NetworkInterface':'97',
        'android/app/Activity;->setContentView':'98',
        'android/content/pm/PackageManager;->clearPackagePreferredActivities':'99',
        'android/app/WallpaperManager;->suggestDesiredDimensions':'100',
        'android/app/Activity;->setPersistent':'101',
        'android/net/wifi/WifiManager;->getConfiguredNetworks':'102',
        'android/net/wifi/WifiManager;->addNetwork':'103',
        'android/telephony/SmsManager;->sendDataMessage':'104',
        'android/location/LocationManager;->getProvider':'105',
        'android/content/ContentResolver;->openOutputStream':'106',
        'android/app/WallpaperManager;->setStream':'107',
        'android/webkit/WebChromeClient;->onGeolocationPermissionsShowPrompt':'108',
        'android/bluetooth/BluetoothAdapter;->getAddress':'109',
        'android/app/KeyguardManager$KeyguardLock;->reenableKeyguard':'110',
        'android/media/AudioManager;->setMicrophoneMute':'111',
        'android/bluetooth/BluetoothAdapter;->getBondedDevices':'112',
        'android/provider/Browser;->getAllBookmarks':'113',
        'android/provider/Browser;->clearSearches':'114',
        'android/net/wifi/WifiManager;->getDhcpInfo':'115',
        'android/telephony/gsm/SmsManager;->sendDataMessage':'116',
        'android/net/wifi/WifiManager;->getScanResults':'117',
        'android/provider/ContactsContract$Contacts;->openContactPhotoInputStream':'118',
        'android/app/Activity;->sendBroadcast':'119',
        'android/bluetooth/BluetoothAdapter;->isDiscovering':'120',
        'android/telephony/SmsManager;->sendMultipartTextMessage':'121',
        'android/location/LocationManager;->addTestProvider':'122',
        'android/provider/Contacts$People;->loadContactPhoto':'123',
        'android/bluetooth/BluetoothAdapter;->setName':'124',
        'android/bluetooth/BluetoothAdapter;->getName':'125',
        'android/app/ActivityManager;->getRecentTasks':'126',
        'android/accounts/AccountManager;->setUserData':'127',
        'android/provider/Contacts$People;->setPhotoData':'128',
        'android/provider/ContactsContract$RawContacts;->getContactLookupUri':'129',
        'android/telephony/TelephonyManager;->getVoiceMailNumber':'130',
        'android/net/wifi/WifiManager;->pingSupplicant':'131',
        'android/provider/Browser;->getAllVisitedUrls':'132',
        'android/provider/Contacts$People;->addToGroup':'133',
        'android/content/pm/PackageManager;->addPreferredActivity':'134',
        'android/media/AsyncPlayer;->play':'135',
        'android/bluetooth/BluetoothAdapter;->isEnabled':'136',
        'android/bluetooth/BluetoothAdapter;->listenUsingRfcommWithServiceRecord':'137',
        'android/provider/ContactsContract$Contacts;->lookupContact':'138',
        'android/app/KeyguardManager;->exitKeyguardSecurely':'139',
        'android/provider/Browser;->clearHistory':'140',
        'android/content/pm/PackageManager;->getPackageSizeInfo':'141',
        'android/media/AudioManager;->setSpeakerphoneOn':'142',
        'android/net/ConnectivityManager;->setNetworkPreference':'143',
        'android/app/Service;->sendBroadcast':'144',
        'android/bluetooth/BluetoothDevice;->getName':'145',
        'android/net/wifi/WifiManager;->disconnect':'146',
        'android/bluetooth/BluetoothDevice;->getBondState':'147',
        'android/net/wifi/WifiManager$MulticastLock;->acquire':'148',
        'android/provider/Telephony$Threads;->getOrCreateThreadId':'149',
        'android/provider/Telephony$Sms;->addMessageToUri':'150',
        'android/media/AudioManager;->setBluetoothScoOn':'151',
        'android/telephony/TelephonyManager;->getDeviceSoftwareVersion':'152',
        'android/media/MediaRecorder;->setVideoSource':'153',
        'android/net/NetworkInfo;->isConnectedOrConnecting':'154',
        'android/accounts/AccountManager;->removeAccount':'155',
        'android/app/StatusBarManager;->expand':'156',
        'android/app/ActivityManagerNative;->forceStopPackage':'157',
        'android/app/ActivityManagerNative;->clearApplicationUserData':'158',
        'android/appwidget/AppWidgetManager;->bindAppWidgetId':'159',
        'android/bluetooth/BluetoothAdapter;->getState':'160',
        'android/net/ConnectivityManager;->setMobileDataEnabled':'161',
        'android/content/Context;->sendStickyBroadcast':'162',
        'android/net/wifi/WifiManager;->saveConfiguration':'163',
        'android/bluetooth/BluetoothAdapter;->startDiscovery':'164',
        'android/net/wifi/WifiManager;->reconnect':'165',
        'android/app/Instrumentation;->sendPointerSync':'166',
        'android/content/ContextWrapper;->startActivity':'167',
        'android/app/Instrumentation;->invokeContextMenuAction':'168',
        'android/app/WallpaperManager;->clear':'169',
        'android/app/Activity;->sendStickyBroadcast':'170',
        'android/provider/Browser;->deleteFromHistory':'171',
        'android/provider/Telephony$Sms$Outbox;->addMessage':'172',
        'android/accounts/AccountManager;->clearPassword':'173',
        'android/accounts/AccountManager;->getPassword':'174',
        'android/provider/Browser;->canClearHistory':'175',
        'android/provider/Calendar$Calendars;->delete':'176',
        'android/app/admin/DevicePolicyManager;->reportFailedPasswordAttempt':'177',
        'android/app/AlarmManager;->setTimeZone':'178',
        'android/provider/Calendar$CalendarAlerts;->query':'179',
        'android/bluetooth/BluetoothAdapter;->setScanMode':'180',
        'android/provider/Settings$Secure;->setLocationProviderEnabled':'181',
        'android/view/Surface;->closeTransaction':'182',
        'android/net/Downloads$ByUri;->getCurrentOtaDownloads':'183',
        'android/accounts/AccountManagerService;->editProperties':'184',
        'android/media/MediaPlayer;->stayAwake':'185',
        'android/app/AlarmManager;->setTime':'186',
        'android/app/Instrumentation;->sendKeyDownUpSync':'187',
        'android/accounts/AbstractAccountAuthenticator;->checkBinderPermission':'188',
        'android/bluetooth/BluetoothHeadset;->startVoiceRecognition':'189',
        'android/provider/CallLog$Calls;->removeExpiredEntries':'190',
        'android/accounts/AccountManagerService;->checkAuthenticateAccountsPermission':'191',
        'android/hardware/Camera;->native_setup':'192',
        'android/telephony/SmsManager;->copyMessageToIcc':'193',
        'android/app/ActivityManagerNative;->getRecentTasks':'194',
        'android/server/search/Searchables;->buildSearchableList':'195',
        'android/server/BluetoothA2dpService;->connectSink':'196',
        'android/app/StatusBarManager;->collapse':'197',
        'android/content/ContentResolver;->getIsSyncable':'198',
        'android/content/ContentResolver;->setIsSyncable':'199',
        'android/os/PowerManager;->reboot':'200',
        'android/widget/QuickContactBadge;->assignContactFromPhone':'201',
        'android/accounts/AccountManager;->blockingGetAuthToken':'202',
        'android/app/ListActivity;->setWallpaper':'203',
        'android/location/LocationManager;->setTestProviderLocation':'204',
        'android/telephony/PhoneNumberUtils;->getNumberFromIntent':'205',
        'android/accounts/AccountManager;->getUserData':'206',
        'android/content/ContentResolver;->addPeriodicSync':'207',
        'android/view/Surface;->setOrientation':'208',
        'android/webkit/WebSettings;->setBlockNetworkLoads':'209',
        'android/accounts/AbstractAccountAuthenticator;->getAccountRemovalAllowed':'210',
        'android/content/ContentResolver;->isSyncActive':'211',
        'android/media/AudioManager;->isWiredHeadsetOn':'212',
        'com/android/http/multipart/Part;->send':'213',
        'android/provider/Browser;->deleteHistoryTimeFrame':'214',
        'android/app/Service;->startService':'215',
        'android/net/wifi/WifiManager;->enableNetwork':'216',
        'android/provider/Telephony$Sms$Sent;->addMessage':'217',
        'android/app/Service;->startActivity':'218',
        'android/net/wifi/WifiManager;->disableNetwork':'219',
        'android/net/ConnectivityManager;->stopUsingNetworkFeature':'220',
        'android/content/ContextWrapper;->startService':'221',
        'android/location/LocationManager;->removeTestProvider':'222',
        'android/content/pm/PackageManager;->installPackage':'223',
        'android/location/LocationManager;->setTestProviderEnabled':'224',
        'android/content/Context;->sendOrderedBroadcast':'225',
        'android/net/wifi/WifiManager;->reassociate':'226',
        'com/android/server/WallpaperManagerService;->bindWallpaperComponentLocked':'227',
        'android/media/AudioManager;->setParameter':'228',
        'android/app/ActivityManagerNative;->restartPackage':'229',
        'android/location/LocationManager;->_requestLocationUpdates':'230',
        'android/bluetooth/BluetoothA2dp;->getSinkState':'231',
        'android/app/StatusBarManager;->disable':'232',
        'android/content/pm/PackageManager;->replacePreferredActivity':'233',
        'android/speech/SpeechRecognizer;->handleCancelMessage':'234',
        'android/location/LocationManager;->addProximityAlert':'235',
        'android/provider/Contacts$People;->markAsContacted':'236',
        'android/provider/ContactsContract$Contacts;->getLookupUri':'237',
        'android/bluetooth/BluetoothAdapter;->enable':'238',
        'android/provider/Contacts$Settings;->setSetting':'239',
        'android/bluetooth/BluetoothSocket;->connect':'240',
        'android/bluetooth/ScoSocket;->acquireWakeLock':'241',
        'android/app/Activity;->sendOrderedBroadcast':'242',
        'android/content/pm/PackageManager;->setApplicationEnabledSetting':'243',
        'android/provider/ContactsContract$Contacts;->markAsContacted':'244',
        'android/app/Application;->setWallpaper':'245',
        'android/provider/Contacts$People;->openContactPhotoInputStream':'246',
        'android/provider/Browser;->updateVisitedHistory':'247',
        'android/content/pm/PackageManager;->deletePackage':'248',
        'android/speech/SpeechRecognizer;->startListening':'249',
        'android/provider/Settings$Secure;->putInt':'250',
        'android/media/AsyncPlayer;->stop':'251',
        'android/app/Activity;->clearWallpaper':'252',
        'android/accounts/AccountManager;->updateCredentials':'253',
        'java/net/URLConnection;->connect':'254',
        'android/content/ContentResolver;->isSyncPending':'255',
        'android/provider/Telephony$Sms;->moveMessageToFolder':'256',
        'android/content/ContextWrapper;->sendBroadcast':'257',
        'java/net/MulticastSocket':'258',
        'android/accounts/AccountManager;->setAuthToken':'259',
        'android/app/Instrumentation;->sendCharacterSync':'260',
        'android/location/LocationManager;->clearTestProviderEnabled':'261',
        'android/app/Service;->removeStickyBroadcast':'262',
        'android/net/wifi/WifiManager;->removeNetwork':'263',
        'android/app/Activity;->removeStickyBroadcast':'264',
        'android/provider/Settings$System;->putFloat':'265',
        'android/provider/Browser;->requestAllIcons':'266',
        'android/net/wifi/WifiManager;->setWifiApEnabled':'267',
        'android/location/LocationManager;->clearTestProviderLocation':'268',
        'android/app/Service;->setWallpaper':'269',
        'android/bluetooth/BluetoothDevice;->createBond':'270',
        'android/location/LocationManager;->addNmeaListener':'271',
        'android/provider/Settings$System;->putLong':'272',
        'android/net/ConnectivityManager;->getNetworkPreference':'273',
        'android/provider/Calendar$Events;->query':'274',
        'android/content/pm/PackageManager;->freeStorage':'275',
        'android/accounts/AccountManager;->peekAuthToken':'276',
        'android/content/ContextWrapper;->setWallpaper':'277',
        'android/accounts/AccountManager;->confirmCredentials':'278',
        'android/app/ActivityManagerNative;->killBackgroundProcesses':'279',
        'android/speech/SpeechRecognizer;->stopListening':'280',
        'android/telephony/TelephonyManager;->getVoiceMailAlphaTag':'281',
        'android/net/ConnectivityManager;->getTetherableUsbRegexs':'282',
        'android/location/LocationManager;->sendExtraCommand':'283',
        'android/view/Surface;->freezeDisplay':'284',
        'android/net/ConnectivityManager;->getMobileDataEnabled':'285',
        'android/view/SurfaceSession':'286',
        'android/provider/Contacts$People;->addToMyContactsGroup':'287',
        }
