CLASS z_humaneval_011_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS
  CREATE PUBLIC.

  PRIVATE SECTION.
    METHODS: test_xor_case1 FOR TESTING,
      test_xor_case2 FOR TESTING,
      test_xor_case3 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_011_test IMPLEMENTATION.

  METHOD test_xor_case1.
    DATA(lv_result) = z_humaneval_011=>string_xor( a = '111000' b = '101010' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '010010' ).
  ENDMETHOD.

  METHOD test_xor_case2.
    DATA(lv_result) = z_humaneval_011=>string_xor( a = '1' b = '1' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '0' ).
  ENDMETHOD.

  METHOD test_xor_case3.
    DATA(lv_result) = z_humaneval_011=>string_xor( a = '0101' b = '0000' ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = '0101' ).
  ENDMETHOD.

ENDCLASS.