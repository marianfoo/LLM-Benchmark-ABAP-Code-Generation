CLASS ltcl_triangle_area_test DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING.

ENDCLASS.

CLASS ltcl_triangle_area_test IMPLEMENTATION.

  METHOD test_case_1.
    DATA(result) = z_humaneval_071=>triangle_area( a = 3 b = 4 c = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '6.00' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA(result) = z_humaneval_071=>triangle_area( a = 1 b = 2 c = 10 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA(result) = z_humaneval_071=>triangle_area( a = 4 b = 8 c = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '8.18' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA(result) = z_humaneval_071=>triangle_area( a = 2 b = 2 c = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '1.73' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA(result) = z_humaneval_071=>triangle_area( a = 1 b = 2 c = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA(result) = z_humaneval_071=>triangle_area( a = 10 b = 5 c = 7 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '16.25' ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA(result) = z_humaneval_071=>triangle_area( a = 2 b = 6 c = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA(result) = z_humaneval_071=>triangle_area( a = 1 b = 1 c = 1 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '0.43' ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA(result) = z_humaneval_071=>triangle_area( a = 2 b = 2 c = 10 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = -1 ).
  ENDMETHOD.

ENDCLASS.