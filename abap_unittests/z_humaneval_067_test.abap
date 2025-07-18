CLASS ltc_fruit_distribution DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING.

ENDCLASS.

CLASS ltc_fruit_distribution IMPLEMENTATION.

  METHOD test_case_1.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '5 apples and 6 oranges' n = 19 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 8 ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '5 apples and 6 oranges' n = 21 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 10 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '0 apples and 1 oranges' n = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '1 apples and 0 oranges' n = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 2 ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '2 apples and 3 oranges' n = 100 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 95 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '2 apples and 3 oranges' n = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 0 ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA(result) = z_humaneval_067=>fruit_distribution( s = '1 apples and 100 oranges' n = 120 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 19 ).
  ENDMETHOD.

ENDCLASS.