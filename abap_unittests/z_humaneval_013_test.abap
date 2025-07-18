CLASS z_humaneval_013_test DEFINITION
  FOR TESTING
  FINAL
  CREATE PUBLIC
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING.

    METHODS check
      IMPORTING
        iv_a        TYPE i
        iv_b        TYPE i
        iv_expected TYPE i.
ENDCLASS.

CLASS z_humaneval_013_test IMPLEMENTATION.

  METHOD check.
    DATA(lv_result) = z_humaneval_013=>greatest_common_divisor( a = iv_a b = iv_b ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = iv_expected ).
  ENDMETHOD.

  METHOD test_case_1.
    check( iv_a = 3 iv_b = 7 iv_expected = 1 ).
  ENDMETHOD.

  METHOD test_case_2.
    check( iv_a = 10 iv_b = 15 iv_expected = 5 ).
  ENDMETHOD.

  METHOD test_case_3.
    check( iv_a = 49 iv_b = 14 iv_expected = 7 ).
  ENDMETHOD.

  METHOD test_case_4.
    check( iv_a = 144 iv_b = 60 iv_expected = 12 ).
  ENDMETHOD.

ENDCLASS.