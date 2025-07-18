CLASS ltcl_test_encrypt DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_encrypt IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'hi' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'lm'
      msg = 'This prints if this assert fails 1 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'asdfghjkl' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'ewhjklnop'
      msg = 'This prints if this assert fails 2 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'gf' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'kj'
      msg = 'This prints if this assert fails 3 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'et' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'ix'
      msg = 'This prints if this assert fails 4 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'faewfawefaewg' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'jeiajeaijeiak'
      msg = 'This prints if this assert fails 5 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'hellomyfriend' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'lippsqcjvmirh'
      msg = 'This prints if this assert fails 6 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'dxzdlmnilfuhmilufhlihufnmlimnufhlimnufhfucufh' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'hbdhpqrmpjylqmpyjlpmlyjrqpmqryjlpmqryjljygyjl'
      msg = 'This prints if this assert fails 7 (good for debugging!)' ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lv_result TYPE string.

    lv_result = z_humaneval_089=>encrypt( 'a' ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 'e'
      msg = 'This prints if this assert fails 8 (also good for debugging!)' ).
  ENDMETHOD.

ENDCLASS.