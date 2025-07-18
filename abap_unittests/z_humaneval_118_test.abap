CLASS ltcl_test_z_humaneval_118 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_yogurt FOR TESTING,
      test_full FOR TESTING,
      test_easy FOR TESTING,
      test_easy_mixed_case FOR TESTING,
      test_ali FOR TESTING,
      test_bad FOR TESTING,
      test_most FOR TESTING,
      test_ab FOR TESTING,
      test_ba FOR TESTING,
      test_quick FOR TESTING,
      test_anime FOR TESTING,
      test_asia FOR TESTING,
      test_above FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_118 IMPLEMENTATION.

  METHOD test_yogurt.
    DATA(lv_result) = z_humaneval_118=>solve( 'yogurt' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'u' ).
  ENDMETHOD.

  METHOD test_full.
    DATA(lv_result) = z_humaneval_118=>solve( 'full' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'u' ).
  ENDMETHOD.

  METHOD test_easy.
    DATA(lv_result) = z_humaneval_118=>solve( 'easy' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_easy_mixed_case.
    DATA(lv_result) = z_humaneval_118=>solve( 'eAsy' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_ali.
    DATA(lv_result) = z_humaneval_118=>solve( 'ali' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_bad.
    DATA(lv_result) = z_humaneval_118=>solve( 'bad' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'a' ).
  ENDMETHOD.

  METHOD test_most.
    DATA(lv_result) = z_humaneval_118=>solve( 'most' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'o' ).
  ENDMETHOD.

  METHOD test_ab.
    DATA(lv_result) = z_humaneval_118=>solve( 'ab' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_ba.
    DATA(lv_result) = z_humaneval_118=>solve( 'ba' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_quick.
    DATA(lv_result) = z_humaneval_118=>solve( 'quick' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_anime.
    DATA(lv_result) = z_humaneval_118=>solve( 'anime' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'i' ).
  ENDMETHOD.

  METHOD test_asia.
    DATA(lv_result) = z_humaneval_118=>solve( 'Asia' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '' ).
  ENDMETHOD.

  METHOD test_above.
    DATA(lv_result) = z_humaneval_118=>solve( 'Above' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'o' ).
  ENDMETHOD.

ENDCLASS.