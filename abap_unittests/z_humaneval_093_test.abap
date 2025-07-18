CLASS ltcl_test_z_humaneval_093 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_z_humaneval_093 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_093=>encode( 'TEST' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'tgst' msg = 'This prints if this assert fails 1 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_093=>encode( 'Mudasir' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'mWDCSKR' msg = 'This prints if this assert fails 2 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_093=>encode( 'YES' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'ygs' msg = 'This prints if this assert fails 3 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_093=>encode( 'This is a message' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'tHKS KS C MGSSCGG' msg = 'This prints if this assert fails 4 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_093=>encode( 'I DoNt KnOw WhAt tO WrItE' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 'k dQnT kNqW wHcT Tq wRkTg' msg = 'This prints if this assert fails 5 (good for debugging!)' ).
  ENDMETHOD.

ENDCLASS.