import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Modal, TouchableOpacity, ScrollView, ActivityIndicator, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getUserProfilePublic } from '../services/api';

interface ProfileViewerModalProps {
    visible: boolean;
    userId: number | null;
    onClose: () => void;
}

const ProfileViewerModal: React.FC<ProfileViewerModalProps> = ({ visible, userId, onClose }) => {
    const [profile, setProfile] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (visible && userId) {
            fetchProfile();
        } else {
            setProfile(null);
            setLoading(true);
        }
    }, [visible, userId]);

    const fetchProfile = async () => {
        setLoading(true);
        try {
            if (userId) {
                const data = await getUserProfilePublic(userId);
                setProfile(data);
            }
        } catch (error) {
            console.error("Failed to fetch public profile", error);
        } finally {
            setLoading(false);
        }
    };

    if (!visible) return null;

    return (
        <Modal
            animationType="slide"
            transparent={true}
            visible={visible}
            onRequestClose={onClose}
        >
            <View style={styles.modalOverlay}>
                <View style={styles.modalContent}>
                    <View style={styles.header}>
                        <TouchableOpacity onPress={onClose} style={styles.closeBtn}>
                            <Ionicons name="close" size={24} color="#333" />
                        </TouchableOpacity>
                        <Text style={styles.headerTitle}>Profile</Text>
                        <View style={{ width: 24 }} />
                    </View>

                    {loading ? (
                        <View style={styles.center}>
                            <ActivityIndicator size="large" color="#FF3366" />
                        </View>
                    ) : profile ? (
                        <ScrollView contentContainerStyle={styles.scrollContent}>
                            <View style={styles.avatarContainer}>
                                <View style={styles.avatar}>
                                    <Text style={styles.avatarText}>
                                        {profile.username?.charAt(0).toUpperCase()}
                                    </Text>
                                </View>
                                <Text style={styles.username}>{profile.username}</Text>
                                {profile.match_reason && (
                                    <Text style={styles.matchReason}>{profile.match_reason}</Text>
                                )}
                            </View>

                            {profile.bio && (
                                <View style={styles.section}>
                                    <Text style={styles.sectionTitle}>About</Text>
                                    <Text style={styles.bioText}>{profile.bio}</Text>
                                </View>
                            )}

                            {profile.tags && profile.tags.length > 0 && (
                                <View style={styles.section}>
                                    <Text style={styles.sectionTitle}>Interests</Text>
                                    <View style={styles.tagContainer}>
                                        {profile.tags.map((tag: string, index: number) => (
                                            <View key={index} style={styles.tag}>
                                                <Text style={styles.tagText}>{tag}</Text>
                                            </View>
                                        ))}
                                    </View>
                                </View>
                            )}

                            {profile.common_movies && profile.common_movies.length > 0 && (
                                <View style={styles.section}>
                                    <Text style={styles.sectionTitle}>You Both Like</Text>
                                    {profile.common_movies.map((movie: string, index: number) => (
                                        <Text key={index} style={styles.infoText}>• {movie}</Text>
                                    ))}
                                </View>
                            )}

                        </ScrollView>
                    ) : (
                        <View style={styles.center}>
                            <Text>Failed to load profile.</Text>
                        </View>
                    )}
                </View>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'flex-end',
    },
    modalContent: {
        backgroundColor: '#fff',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        height: '80%',
        paddingBottom: 20,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333',
    },
    closeBtn: {
        padding: 5,
    },
    center: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    scrollContent: {
        padding: 20,
    },
    avatarContainer: {
        alignItems: 'center',
        marginBottom: 25,
    },
    avatar: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 10,
    },
    avatarText: {
        fontSize: 40,
        fontWeight: 'bold',
        color: '#666',
    },
    username: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#333',
    },
    matchReason: {
        fontSize: 14,
        color: '#FF3366',
        marginTop: 5,
        fontStyle: 'italic',
    },
    section: {
        marginBottom: 20,
    },
    sectionTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#333',
        marginBottom: 8,
    },
    bioText: {
        fontSize: 15,
        color: '#555',
        lineHeight: 22,
    },
    infoText: {
        fontSize: 15,
        color: '#444',
        marginBottom: 4,
    },
    tagContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
    },
    tag: {
        backgroundColor: '#f0f0f0',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
        marginRight: 8,
        marginBottom: 8,
    },
    tagText: {
        fontSize: 12,
        color: '#555',
    },
});

export default ProfileViewerModal;
